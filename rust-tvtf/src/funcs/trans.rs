use anyhow::{Context, Result, anyhow};
use arrow::array::{ArrayRef, AsArray, ListBuilder, StringBuilder};
use arrow::datatypes::{Float64Type, Int64Type, TimestampMicrosecondType};
use arrow::{
    array::{Array, BooleanArray, Float64Array, Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use chrono::{DateTime, TimeZone, Utc};
use hashbrown::HashMap;
use parking_lot::Mutex;
use regex::Regex;
use rust_tvtf_api::TableFunction;
use rust_tvtf_api::arg::{Arg, Args};
use smallvec::{SmallVec, smallvec};
use std::sync::LazyLock;
use std::{collections::VecDeque, sync::Arc, time::Duration};

// Reserved field names for transaction events and raw events
const FIELD_TIME: &str = "_time";
const FIELD_MESSAGE: &str = "_message";
const FIELD_DURATION: &str = "_duration";
const FIELD_EVENT_COUNT: &str = "_event_count";
const FIELD_IS_CLOSED: &str = "_is_closed";

// Trans specific fields that are not part of the original event's fields
const TRANS_RESERVED_FIELDS: [&str; 5] = [
    FIELD_TIME,
    FIELD_MESSAGE,
    FIELD_DURATION,
    FIELD_EVENT_COUNT,
    FIELD_IS_CLOSED,
];

#[derive(Debug, Clone)]
pub struct TransParams {
    pub fields: SmallVec<[String; 4]>,
    pub allow_nulls: bool,
    pub starts_with: Option<String>,
    pub starts_with_regex: Option<Regex>,
    pub starts_if_field: Option<String>,
    pub ends_with: Option<String>,
    pub ends_with_regex: Option<Regex>,
    pub ends_if_field: Option<String>,
    pub max_span: Duration,
    pub max_events: u64,
}

impl Default for TransParams {
    fn default() -> Self {
        Self {
            fields: Default::default(),
            starts_with: Default::default(),
            starts_with_regex: Default::default(),
            starts_if_field: Default::default(),
            ends_with: Default::default(),
            ends_with_regex: Default::default(),
            ends_if_field: Default::default(),
            max_span: Default::default(),
            max_events: 1000,
            allow_nulls: false,
        }
    }
}

impl TransParams {
    pub fn new(params: Option<Args>, named_arguments: Vec<(String, Arg)>) -> Result<Self> {
        let mut parsed_params = TransParams {
            max_events: 1000,
            ..Default::default()
        };
        if let Some(params_vec_raw) = params {
            let mut params_iter = params_vec_raw.into_iter().filter(|x| x.is_scalar());
            if let Some(_first_param) = params_iter.next() {
                return Err(anyhow!("Too many positional parameters."));
            }
        }

        for (name, arg) in named_arguments {
            match name.as_str() {
                "fields" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {name}. Expected string."));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.fields = s.split(',').map(|f| f.trim().to_string()).collect();
                }
                "starts_with" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {name}. Expected string."));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.starts_with = Some(s);
                }
                "starts_with_regex" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {name}. Expected string."));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.starts_with_regex =
                        Some(Regex::new(&s).context(format!("Invalid regex for {name}: {s}"))?);
                }
                "starts_if_field" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {name}. Expected string."));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.starts_if_field = Some(s);
                }
                "ends_with" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {name}. Expected string."));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.ends_with = Some(s);
                }
                "ends_with_regex" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid typ5e for {name}. Expected string."));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.ends_with_regex =
                        Some(Regex::new(&s).context(format!("Invalid regex for {name}: {s}"))?);
                }
                "ends_if_field" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {name}. Expected string."));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.ends_if_field = Some(s);
                }
                "max_span" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {name}. Expected string."));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    static DURATION_REGEX: LazyLock<Regex> =
                        LazyLock::new(|| Regex::new(r"(\d+)([smhd])").unwrap());
                    if let Some(m) = DURATION_REGEX.captures(&s) {
                        let value: u64 = m[1].parse()?;
                        let unit = &m[2];
                        let seconds = match unit {
                            "s" => value,
                            "m" => value * 60,
                            "h" => value * 3600,
                            "d" => value * 86400,
                            _ => return Err(anyhow!("Invalid time unit: {unit}")),
                        };
                        parsed_params.max_span = Duration::from_secs(seconds);
                    } else {
                        return Err(anyhow!("Invalid max_span format: {s}"));
                    }
                }
                "max_events" => {
                    let max_events_val: i64 = match arg {
                        Arg::Int(i) => i,
                        _ => {
                            return Err(anyhow!("Invalid type for {name}. Expected integer."));
                        }
                    };
                    parsed_params.max_events = if max_events_val <= 0 {
                        1000
                    } else {
                        max_events_val as u64
                    };
                }
                _ => {
                    return Err(anyhow!(
                        "Invalid named parameter for trans table function: {name}"
                    ));
                }
            }
        }
        Ok(parsed_params)
    }

    fn check_boolean_field_condition(
        &self,
        event: &HashMap<String, SmallVec<[String; 4]>>,
        field_name_opt: &Option<String>,
    ) -> bool {
        if let Some(field_name) = field_name_opt
            && let Some(value) = event.get(field_name)
        {
            return value.iter().all(|v| v.eq_ignore_ascii_case("true"));
        }
        false
    }

    fn matches_starts_with(&self, event: &HashMap<String, SmallVec<[String; 4]>>) -> bool {
        let Some(message) = event.get(FIELD_MESSAGE) else {
            return false;
        };

        if let Some(s) = &self.starts_with
            && message.iter().any(|v| v.contains(s))
        {
            return true;
        }
        if let Some(r) = &self.starts_with_regex
            && message.iter().any(|v| r.is_match(v))
        {
            return true;
        }
        // Check ends_if_field condition
        if self.check_boolean_field_condition(event, &self.starts_if_field) {
            return true;
        }
        false
    }

    fn matches_ends_with(&self, event: &HashMap<String, SmallVec<[String; 4]>>) -> bool {
        let Some(message) = event.get(FIELD_MESSAGE) else {
            return false;
        };

        if let Some(s) = &self.ends_with
            && message.iter().any(|v| v.contains(s))
        {
            return true;
        }
        if let Some(r) = &self.ends_with_regex
            && message.iter().any(|v| r.is_match(v))
        {
            return true;
        }
        // Check ends_if_field condition
        if self.check_boolean_field_condition(event, &self.ends_if_field) {
            return true;
        }
        false
    }
}

#[derive(Debug, Clone, PartialEq)]
enum TransactionType {
    Start,
    End,
    Normal,
}

#[derive(Debug, Clone)]
struct Transaction {
    field_names: SmallVec<[String; 4]>,
    fields: HashMap<String, SmallVec<[String; 4]>>,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    messages: VecDeque<SmallVec<[String; 4]>>,
    event_count: u64,
    is_closed: bool,
    transaction_type: TransactionType,
}

impl Transaction {
    fn new(field_names: SmallVec<[String; 4]>, transaction_type: TransactionType) -> Self {
        Transaction {
            field_names,
            fields: HashMap::new(),
            start_time: None,
            end_time: None,
            messages: VecDeque::new(),
            event_count: 0,
            is_closed: false,
            transaction_type,
        }
    }

    fn merge_event(&mut self, event: &HashMap<String, SmallVec<[String; 4]>>) {
        if let Some(message) = event.get(FIELD_MESSAGE) {
            if message.is_empty() {
                return;
            }
            self.messages.push_front(message.clone());
        }

        for (k, v) in event {
            if TRANS_RESERVED_FIELDS.contains(&k.as_str()) {
                continue;
            }
            if self.field_names.contains(k) {
                self.fields.insert(k.clone(), v.clone());
            } else {
                let entry = self.fields.entry(k.clone()).or_default();
                entry.extend(v.clone());
            }
        }
    }

    fn add_event(&mut self, event: &HashMap<String, SmallVec<[String; 4]>>) -> Result<()> {
        let time_str = event
            .get(FIELD_TIME)
            .context("Event missing _time field or _time is null")?;

        let time = time_str
            .first()
            .context("No _time in transaction")?
            .parse::<i64>()
            .map(|ts| Utc.timestamp_micros(ts).earliest().unwrap_or_else(Utc::now))
            .context("Failed to parse _time as timestamp")?;

        self.event_count += 1;

        if self.end_time.is_none() {
            self.start_time = Some(time);
            self.end_time = Some(time);
            self.merge_event(event);
        } else if time <= self.start_time.unwrap() {
            self.start_time = Some(time);
            self.merge_event(event);
        } else if time <= self.end_time.unwrap() && time >= self.start_time.unwrap() {
            self.merge_event(event);
        }

        Ok(())
    }

    fn get_duration(&self) -> Option<f64> {
        let (Some(start), Some(end)) = (self.start_time, self.end_time) else {
            return None;
        };
        if let Some(microsecond) = (end - start).num_microseconds() {
            Some(microsecond as f64 / 1000000.0)
        } else {
            Some((end - start).num_milliseconds() as f64 / 1000.0)
        }
    }

    fn get_event_count(&self) -> u64 {
        self.event_count
    }

    fn set_is_closed(&mut self) {
        self.is_closed = true;
    }

    #[allow(dead_code)]
    fn get_is_closed(&self) -> bool {
        self.is_closed
    }
}

fn to_record_batch(
    params: &TransParams,
    transactions: &[Transaction],
) -> Result<Option<RecordBatch>> {
    if transactions.is_empty() {
        return Ok(None);
    }

    // Collect all unique field names from all transactions
    let mut all_field_names = std::collections::HashSet::new();
    for transaction in transactions {
        for field_name in transaction.fields.keys() {
            all_field_names.insert(field_name.clone());
        }
    }
    let mut sorted_field_names: Vec<String> = all_field_names.into_iter().collect();
    sorted_field_names.sort();

    // Create schema
    let mut fields = vec![
        Field::new(
            FIELD_TIME,
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            FIELD_MESSAGE,
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
        Field::new(FIELD_DURATION, DataType::Float64, true),
        Field::new(FIELD_EVENT_COUNT, DataType::Int64, true),
        Field::new(FIELD_IS_CLOSED, DataType::Boolean, true),
    ];

    for field_name in &sorted_field_names {
        if params.fields.contains(field_name) {
            fields.push(Field::new(field_name, DataType::Utf8, true));
        } else {
            fields.push(Field::new(
                field_name,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ));
        }
    }

    // Create arrays
    let mut arrays: Vec<Arc<dyn Array>> = Vec::new();

    // Time array
    let time_values: Vec<Option<i64>> = transactions
        .iter()
        .map(|t| t.start_time.map(|dt| dt.timestamp_micros()))
        .collect();
    arrays.push(Arc::new(TimestampMicrosecondArray::from(time_values)));

    // Messages array
    let mut messages_builder = ListBuilder::new(StringBuilder::new());
    for transaction in transactions {
        for messages in transaction.messages.iter() {
            for message in messages {
                messages_builder.values().append_value(message);
            }
        }
        messages_builder.append(true);
    }
    arrays.push(Arc::new(messages_builder.finish()));

    // Duration array
    let duration_values: Vec<Option<f64>> = transactions.iter().map(|t| t.get_duration()).collect();
    arrays.push(Arc::new(Float64Array::from(duration_values)));

    // Event count array
    let event_count_values: Vec<i64> = transactions
        .iter()
        .map(|t| t.get_event_count() as i64)
        .collect();
    arrays.push(Arc::new(Int64Array::from(event_count_values)));

    // Is closed array
    let is_closed_values: Vec<bool> = transactions.iter().map(|t| t.is_closed).collect();
    arrays.push(Arc::new(BooleanArray::from(is_closed_values)));

    // Dynamic field arrays
    for field_name in &sorted_field_names {
        if params.fields.contains(field_name) {
            let mut field_builder = StringBuilder::new();
            for transaction in transactions.iter() {
                let Some(values) = transaction.fields.get(field_name) else {
                    field_builder.append_null();
                    continue;
                };
                let mut values = values.clone();
                values.sort_unstable();
                values.dedup();
                let Some(value) = values.first() else {
                    field_builder.append_null();
                    continue;
                };
                field_builder.append_value(value);
            }
            arrays.push(Arc::new(field_builder.finish()));
        } else {
            let mut field_builder = ListBuilder::new(StringBuilder::new());
            for transaction in transactions.iter() {
                let Some(values) = transaction.fields.get(field_name) else {
                    field_builder.append_null();
                    continue;
                };
                let mut values = values.clone();
                values.sort_unstable();
                values.dedup();
                if values.is_empty() {
                    field_builder.append_null();
                    continue;
                }
                for value in values {
                    field_builder.values().append_value(value);
                }
                field_builder.append(true);
            }
            arrays.push(Arc::new(field_builder.finish()));
        }
    }

    let schema = Arc::new(Schema::new(fields));
    Ok(Some(
        RecordBatch::try_new(schema, arrays).context("Failed to create record batch")?,
    ))
}

type TransKey = SmallVec<[Option<String>; 4]>;

#[derive(Clone, Debug)]
struct TransactionPool {
    params: TransParams,
    frozen_trans: Vec<Transaction>,
    live_trans: HashMap<TransKey, Transaction>,
    pending_transactions: Vec<(TransKey, Transaction)>,
    earliest_event_timestamp: Option<DateTime<Utc>>,
    trans_complete_flag: HashMap<TransKey, bool>,
}

impl TransactionPool {
    fn new(params: TransParams) -> Self {
        TransactionPool {
            params,
            frozen_trans: Vec::new(),
            live_trans: HashMap::new(),
            pending_transactions: Vec::new(),
            earliest_event_timestamp: None,
            trans_complete_flag: HashMap::new(),
        }
    }

    fn is_valid_event(&self, event: &HashMap<String, SmallVec<[String; 4]>>) -> bool {
        let Some(time_str) = event.get(FIELD_TIME) else {
            return false;
        };
        let Some(time_str_val) = time_str.first() else {
            return false;
        };

        let Ok(time) = time_str_val
            .parse::<i64>()
            .map(|ts| Utc.timestamp_micros(ts).earliest().unwrap_or_else(Utc::now))
        else {
            return false;
        };

        // Check if we have transkey fields defined
        if !self.params.fields.is_empty() {
            // Check if all transkey fields are null/empty
            let all_fields_null = self
                .params
                .fields
                .iter()
                .all(|f| event.get(f).is_none_or(|v| v.is_empty()));

            // If all fields are null, discard the event
            if all_fields_null {
                return false;
            }
        }

        if let Some(earliest_ts) = self.earliest_event_timestamp
            && time > earliest_ts
        {
            return false;
        }
        true
    }

    fn freeze_trans_exceeded_max_span_restriction(&mut self) {
        if self.params.max_span.as_secs() > 0 {
            let mut keys_to_freeze = Vec::new();
            if let Some(earliest_ts) = self.earliest_event_timestamp {
                for (trans_key, trans) in &self.live_trans {
                    if let Some(end_time) = trans.end_time
                        && (end_time - earliest_ts).num_seconds() as u64
                            > self.params.max_span.as_secs()
                    {
                        keys_to_freeze.push(trans_key.clone());
                    }
                }
            }
            for key in keys_to_freeze {
                if let Some(trans) = self.live_trans.remove(&key) {
                    self.frozen_trans.push(trans);
                    self.trans_complete_flag.remove(&key);
                }
            }
        }
    }

    fn make_trans_key(&self, event: &HashMap<String, SmallVec<[String; 4]>>) -> TransKey {
        self.params
            .fields
            .iter()
            .map(|f| {
                event.get(f).and_then(|v_opt| {
                    // Check if the field value is empty (null/empty)
                    if v_opt.is_empty() {
                        // Return None to indicate null value
                        None
                    } else {
                        // Get the first value
                        v_opt.first().as_ref().map(|s| s.to_string())
                    }
                })
            })
            .collect()
    }

    /// Compare two transaction keys, ignoring null values in the comparison
    /// Returns true if the keys match (considering nulls as wildcards)
    fn trans_keys_match(key1: &TransKey, key2: &TransKey) -> bool {
        if key1.len() != key2.len() {
            return false;
        }

        for (val1, val2) in key1.iter().zip(key2.iter()) {
            match (val1, val2) {
                // If either value is None (null), they match
                (None, _) | (_, None) => continue,
                // If both are Some, they must be equal
                (Some(v1), Some(v2)) => {
                    if v1 != v2 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Find a matching transaction key in the live transactions, considering null values as wildcards
    fn find_matching_trans_key(&self, target_key: &TransKey) -> Option<TransKey> {
        self.live_trans
            .keys()
            .find(|&key| Self::trans_keys_match(key, target_key))
            .cloned()
    }

    fn add_event(&mut self, event: HashMap<String, SmallVec<[String; 4]>>) -> Result<()> {
        if !self.is_valid_event(&event) {
            return Ok(());
        }

        let time_str = event.get(FIELD_TIME).unwrap();
        let time = time_str
            .first()
            .context("No _time field")?
            .parse::<i64>()
            .map(|ts| Utc.timestamp_micros(ts).earliest().unwrap_or_else(Utc::now))
            .context("Failed to parse _time as timestamp")?;
        self.earliest_event_timestamp = Some(time);

        self.freeze_trans_exceeded_max_span_restriction();

        let trans_key = self.make_trans_key(&event);

        // Find a matching transaction key, considering null values as wildcards
        let matching_key = self.find_matching_trans_key(&trans_key);
        let current_trans = matching_key
            .as_ref()
            .and_then(|key| self.live_trans.remove(key));

        if current_trans.is_none() {
            let transaction_type = if self.params.matches_starts_with(&event) {
                TransactionType::Start
            } else if self.params.matches_ends_with(&event) {
                TransactionType::End
            } else {
                TransactionType::Normal
            };

            let mut new_trans = Transaction::new(self.params.fields.clone(), transaction_type);
            new_trans.add_event(&event)?;

            if self.params.matches_ends_with(&event)
                || (self.params.ends_with.is_none()
                    && self.params.ends_with_regex.is_none()
                    && self.params.ends_if_field.is_none())
            {
                // Use the original trans_key for the flag, not the matching key
                self.trans_complete_flag.insert(trans_key.clone(), true);
            }

            if self.params.matches_starts_with(&event) {
                // Use the original trans_key for the flag, not the matching key
                if self.trans_complete_flag.contains_key(&trans_key) {
                    new_trans.set_is_closed();
                }
                self.frozen_trans.push(new_trans);
                // Use the original trans_key for the flag, not the matching key
                self.trans_complete_flag.remove(&trans_key);
            } else {
                // Insert with the original trans_key
                self.live_trans.insert(trans_key.clone(), new_trans);
            }
        } else if let Some(mut trans) = current_trans {
            if self.params.matches_starts_with(&event) {
                trans.transaction_type = TransactionType::Start;
                trans.add_event(&event)?;
                // Use the matching key for the flag
                if let Some(ref key) = matching_key
                    && self.trans_complete_flag.contains_key(key)
                {
                    trans.set_is_closed();
                }
                self.frozen_trans.push(trans);
                // Use the matching key for the flag
                if let Some(ref key) = matching_key {
                    self.trans_complete_flag.remove(key);
                }
            } else if self.params.matches_ends_with(&event) {
                trans.transaction_type = TransactionType::End;
                self.frozen_trans.push(trans);

                let mut new_trans =
                    Transaction::new(self.params.fields.clone(), TransactionType::Normal);
                new_trans.add_event(&event)?;
                // Insert with the original trans_key
                self.live_trans.insert(trans_key.clone(), new_trans);
                // Use the original trans_key for the flag
                self.trans_complete_flag.insert(trans_key.clone(), true);
            } else {
                trans.add_event(&event)?;
                // Insert with the original trans_key
                self.live_trans.insert(trans_key.clone(), trans);
            }
        }

        if let Some(trans) = self.live_trans.get(&trans_key)
            && trans.get_event_count() >= self.params.max_events
        {
            let trans_to_freeze = self.live_trans.remove(&trans_key).unwrap();
            self.frozen_trans.push(trans_to_freeze);
            self.trans_complete_flag.remove(&trans_key);
        }

        Ok(())
    }

    /// Post-process transactions to implement bracket matching logic for all start/end conditions
    fn post_process_transactions(&mut self) {
        // Apply bracket matching if we're using any start/end conditions
        let has_start_conditions = self.params.starts_with.is_some()
            || self.params.starts_with_regex.is_some()
            || self.params.starts_if_field.is_some();

        let has_end_conditions = self.params.ends_with.is_some()
            || self.params.ends_with_regex.is_some()
            || self.params.ends_if_field.is_some();

        if !has_start_conditions && !has_end_conditions {
            return;
        }

        // Collect all transactions (frozen and live) for post-processing
        let mut all_transactions: Vec<Transaction> = Vec::new();
        all_transactions.append(&mut self.frozen_trans);
        for (_, trans) in self.live_trans.drain() {
            all_transactions.push(trans);
        }
        // Also drain pending transactions
        for (_, trans) in self.pending_transactions.drain(..) {
            all_transactions.push(trans);
        }

        // Separate start and end transactions, keeping track of their times
        let mut start_transactions: Vec<(DateTime<Utc>, Transaction)> = Vec::new();
        let mut end_transactions: Vec<(DateTime<Utc>, Transaction)> = Vec::new();
        let mut other_transactions: Vec<Transaction> = Vec::new();

        for trans in all_transactions {
            // Skip already closed transactions
            if trans.get_is_closed() {
                other_transactions.push(trans);
                continue;
            }

            let time = trans.start_time.unwrap_or_else(Utc::now);

            match trans.transaction_type {
                TransactionType::Start => {
                    start_transactions.push((time, trans));
                }
                TransactionType::End => {
                    end_transactions.push((time, trans));
                }
                TransactionType::Normal => {
                    other_transactions.push(trans);
                }
            }
        }

        // Sort by time (ascending order - oldest first)
        start_transactions.sort_by(|a, b| a.0.cmp(&b.0));
        end_transactions.sort_by(|a, b| a.0.cmp(&b.0));

        // Implement bracket matching logic
        // For each start, find the earliest end that comes after it and is not yet matched
        let mut matched_transactions: Vec<Transaction> = Vec::new();
        let mut unmatched_ends: std::collections::HashSet<usize> = std::collections::HashSet::new();

        // Initialize all ends as unmatched
        for i in 0..end_transactions.len() {
            unmatched_ends.insert(i);
        }

        // For each start (in time order), match it with the earliest available end
        for (start_time, mut start_trans) in start_transactions {
            // Find the earliest unmatched end that comes after this start
            let mut best_match: Option<usize> = None;
            let mut best_time: Option<DateTime<Utc>> = None;

            for (i, (end_time, _)) in end_transactions.iter().enumerate() {
                if !unmatched_ends.contains(&i) {
                    continue; // Already matched
                }

                if *end_time >= start_time {
                    // This end comes after the start
                    match best_time {
                        None => {
                            // First match found
                            best_match = Some(i);
                            best_time = Some(*end_time);
                        }
                        Some(current_best) => {
                            if *end_time < current_best {
                                // Found an earlier match
                                best_match = Some(i);
                                best_time = Some(*end_time);
                            }
                        }
                    }
                }
            }

            // If we found a match, merge the transactions
            if let Some(match_index) = best_match {
                let (end_time, end_trans) = end_transactions[match_index].clone();
                unmatched_ends.remove(&match_index);

                // Merge the start with the end transaction
                for message in end_trans.messages {
                    // Add to the back to maintain time order
                    start_trans.messages.push_back(message);
                }
                for (field_name, field_values) in end_trans.fields {
                    let entry = start_trans.fields.entry(field_name).or_default();
                    entry.extend(field_values);
                }
                start_trans.event_count += end_trans.event_count;

                // Set the end time
                start_trans.end_time = Some(end_time);

                start_trans.set_is_closed();
                matched_transactions.push(start_trans);
            } else {
                // No match found, keep the start transaction as is
                matched_transactions.push(start_trans);
            }
        }

        // Add any remaining unmatched end transactions
        for (i, (_time, transaction)) in end_transactions.iter().enumerate() {
            if unmatched_ends.contains(&i) {
                matched_transactions.push(transaction.clone());
            }
        }

        // Add other transactions (including already closed ones)
        matched_transactions.extend(other_transactions);

        // Move all processed transactions back to frozen_trans
        self.frozen_trans = matched_transactions;
    }

    fn get_frozen_trans(&mut self) -> Vec<Transaction> {
        self.post_process_transactions();
        std::mem::take(&mut self.frozen_trans)
    }

    fn get_live_trans(&mut self) -> Vec<Transaction> {
        // For live transactions, we don't apply bracket matching
        let mut live_transactions: Vec<Transaction> = Vec::new();
        for (_, trans) in self.live_trans.drain() {
            live_transactions.push(trans);
        }
        // Also drain pending transactions
        for (_, trans) in self.pending_transactions.drain(..) {
            live_transactions.push(trans);
        }
        live_transactions
    }
}

#[derive(Default)]
pub struct TransFunction {
    trans_pool: Option<TransactionPool>,
    mutex: Mutex<()>,
}

impl TransFunction {
    pub fn new(params: Option<Args>, named_arguments: Vec<(String, Arg)>) -> Result<Self> {
        let trans_params = TransParams::new(params, named_arguments)?;
        Ok(TransFunction {
            trans_pool: Some(TransactionPool::new(trans_params)),
            mutex: Mutex::default(),
        })
    }
}

fn scalar_to_string(array: ArrayRef, row_idx: usize) -> Option<String> {
    match array.data_type() {
        DataType::Utf8 => Some(array.as_string::<i32>().value(row_idx).to_string()),
        DataType::Int64 => Some(array.as_primitive::<Int64Type>().value(row_idx).to_string()),
        DataType::Float64 => Some(
            array
                .as_primitive::<Float64Type>()
                .value(row_idx)
                .to_string(),
        ),
        DataType::Boolean => Some(array.as_boolean().value(row_idx).to_string()),
        DataType::Timestamp(_, _) => Some(
            array
                .as_primitive::<TimestampMicrosecondType>()
                .value(row_idx)
                .to_string(),
        ),
        _ => None,
    }
}

fn array_to_strings(array: ArrayRef, row_idx: usize) -> SmallVec<[String; 4]> {
    match array.data_type() {
        DataType::Utf8 => smallvec![scalar_to_string(array, row_idx).expect("utf8 must be scalar")],
        DataType::Int64 => {
            smallvec![scalar_to_string(array, row_idx).expect("int64 must be scalar")]
        }
        DataType::Float64 => {
            smallvec![scalar_to_string(array, row_idx).expect("float64 must be scalar")]
        }
        DataType::Boolean => {
            smallvec![scalar_to_string(array, row_idx).expect("boolean must be scalar")]
        }
        DataType::Timestamp(_, _) => {
            smallvec![scalar_to_string(array, row_idx).expect("timestamp must be scalar")]
        }
        DataType::List(_inner_field) => {
            let nested_list = array.as_list::<i32>().value(row_idx);
            let mut small_vec: SmallVec<[String; 4]> = smallvec![];
            for i in 0..nested_list.len() {
                let inner = array_to_strings(Arc::clone(&nested_list), i);
                small_vec.extend(inner);
            }
            small_vec
        }
        _ => smallvec![],
    }
}

impl TableFunction for TransFunction {
    fn process(&mut self, input: RecordBatch) -> Result<Option<RecordBatch>> {
        let _lock = self.mutex.lock();
        let trans_pool = self
            .trans_pool
            .as_mut()
            .context("TransPool not initialized")?;

        let schema = input.schema();
        let mut events: Vec<HashMap<String, SmallVec<[String; 4]>>> =
            Vec::with_capacity(input.num_rows());

        for row_idx in 0..input.num_rows() {
            let mut event: HashMap<String, SmallVec<[String; 4]>> = HashMap::new();
            for col_idx in 0..input.num_columns() {
                let field = schema.field(col_idx);
                let array = Arc::clone(input.column(col_idx));
                if array.is_null(row_idx) {
                    continue;
                }
                event.insert(field.name().clone(), array_to_strings(array, row_idx));
            }
            events.push(event);
        }

        for event in events {
            trans_pool.add_event(event)?;
        }

        let frozen = trans_pool.get_frozen_trans();
        to_record_batch(&trans_pool.params, &frozen)
    }

    fn finalize(&mut self) -> Result<Option<RecordBatch>> {
        let _lock = self.mutex.lock();
        let trans_pool = self
            .trans_pool
            .as_mut()
            .context("TransPool not initialized")?;

        let lives = trans_pool.get_live_trans();
        to_record_batch(&trans_pool.params, &lives)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Array, BooleanArray, Float64Array, Int64Array, ListArray, StringArray,
        TimestampMicrosecondArray,
    };
    use chrono::{Duration as ChronoDuration, Utc};
    use std::sync::Arc;

    // Helper function to create a simple event HashMap
    fn create_event(
        time_micros: i64,
        message: &str,
        other_fields: &[(&str, &str)],
    ) -> HashMap<String, SmallVec<[String; 4]>> {
        let mut event = HashMap::new();
        event.insert(FIELD_TIME.to_string(), smallvec![time_micros.to_string()]);
        event.insert(FIELD_MESSAGE.to_string(), smallvec![message.to_string()]);
        for (k, v) in other_fields {
            event.insert(k.to_string(), smallvec![v.to_string()]);
        }
        event
    }

    fn create_event_nullable(
        time_micros: i64,
        message: &str,
        other_fields: &[(&str, Option<&str>)],
    ) -> HashMap<String, SmallVec<[String; 4]>> {
        let mut event = HashMap::new();
        event.insert(FIELD_TIME.to_string(), smallvec![time_micros.to_string()]);
        event.insert(FIELD_MESSAGE.to_string(), smallvec![message.to_string()]);
        for (k, v) in other_fields {
            if let Some(v) = *v {
                event.insert(k.to_string(), smallvec![v.to_string()]);
            } else {
                event.insert(k.to_string(), smallvec![]);
            }
        }
        event
    }

    /// Helper to extract values from a ListArray of StringArray.
    ///
    /// Correctly accesses the inner StringArray from the ListArray.
    fn extract_list_string_values(array: &Arc<dyn Array>) -> Vec<String> {
        let list_array = array.as_any().downcast_ref::<ListArray>().unwrap();
        // Get the inner array which contains the strings for the first list entry
        let inner_array = list_array.value(0);
        let values_array = inner_array.as_any().downcast_ref::<StringArray>().unwrap();
        (0..values_array.len())
            .map(|i| values_array.value(i).to_string())
            .collect()
    }

    #[test]
    fn test_trans_params_new() {
        let params_args = Some(vec![]);
        let named_args = vec![
            (
                "fields".to_string(),
                Arg::String("client_ip,session_id".to_string()),
            ),
            ("starts_with".to_string(), Arg::String("login".to_string())),
            (
                "starts_if_field".to_string(),
                Arg::String("is_start_event".to_string()),
            ),
            (
                "ends_with_regex".to_string(),
                Arg::String("logout_\\\\d+".to_string()),
            ),
            (
                "ends_if_field".to_string(),
                Arg::String("is_end_event".to_string()),
            ),
            ("max_span".to_string(), Arg::String("10m".to_string())),
            ("max_events".to_string(), Arg::Int(500)),
        ];

        let params = TransParams::new(params_args, named_args).unwrap();

        assert_eq!(params.fields.to_vec(), vec!["client_ip", "session_id"]);
        assert_eq!(params.starts_with, Some("login".to_string()));
        assert!(params.starts_with_regex.is_none());
        assert_eq!(params.starts_if_field, Some("is_start_event".to_string()));
        assert_eq!(params.ends_with_regex.unwrap().as_str(), "logout_\\\\d+");
        assert_eq!(params.ends_if_field, Some("is_end_event".to_string()));
        assert_eq!(params.max_span, Duration::from_secs(600));
        assert_eq!(params.max_events, 500);

        // Test with no parameters (both scalar and named)
        let default_params = TransParams::new(None, Vec::new()).unwrap();
        assert_eq!(default_params.max_events, 1000);
        assert!(default_params.fields.is_empty());
        assert!(default_params.starts_if_field.is_none());
        assert!(default_params.ends_if_field.is_none());

        // Test error for invalid type for positional 'fields' parameter (e.g., Int instead of String)
        let err_scalar_type = Some(vec![Arg::Int(123)]);
        assert!(TransParams::new(err_scalar_type, Vec::new()).is_err());

        // Test error for too many scalar parameters
        let err_too_many_scalar_params = Some(vec![
            Arg::String("field1,field2".to_string()),
            Arg::String("another_scalar_arg".to_string()),
        ]);
        assert!(TransParams::new(err_too_many_scalar_params, Vec::new()).is_err());

        // Test error for invalid named parameter name
        let err_named_param_name = vec![(
            "unknown_param".to_string(),
            Arg::String("value".to_string()),
        )];
        assert!(TransParams::new(None, err_named_param_name).is_err());

        // Test error for invalid named parameter type (e.g., max_span expects String, pass Int)
        let err_named_param_type = vec![("max_span".to_string(), Arg::Int(10))];
        assert!(TransParams::new(None, err_named_param_type).is_err());

        // Test max_events with Arg::Float
        let named_args_float_max_events = vec![("max_events".to_string(), Arg::Int(750))];
        let params_float_max_events = TransParams::new(None, named_args_float_max_events).unwrap();
        assert_eq!(params_float_max_events.max_events, 750);
    }

    #[test]
    fn test_transaction_add_event_and_merge() {
        let mut trans = Transaction::new(smallvec!["user_id".to_string()], TransactionType::Normal);
        let now = Utc::now();

        let event1 = create_event(
            (now - ChronoDuration::seconds(20)).timestamp_micros(),
            "message two",
            &[("user_id", "user1"), ("status", "success")],
        );
        let event2 = create_event(
            (now - ChronoDuration::seconds(10)).timestamp_micros(),
            "message one",
            &[("user_id", "user1"), ("data", "abc")],
        );
        let event3 = create_event(
            (now - ChronoDuration::seconds(30)).timestamp_micros(),
            "message three",
            &[("user_id", "user1"), ("status", "fail")],
        );

        trans.add_event(&event2).unwrap();
        trans.add_event(&event1).unwrap();
        trans.add_event(&event3).unwrap();

        assert_eq!(trans.get_event_count(), 3);
        assert_eq!(trans.messages.len(), 3);
        assert_eq!(
            trans.messages.front().unwrap().first().unwrap(),
            "message three"
        );
        assert_eq!(
            trans.messages.back().unwrap().first().unwrap(),
            "message one"
        );

        let expected: SmallVec<[&str; 4]> = smallvec!["user1"];
        assert_eq!(trans.fields["user_id"], expected);
        assert!(trans.fields["status"].contains(&"success".to_string()));
        assert!(trans.fields["status"].contains(&"fail".to_string()));
        let expected: SmallVec<[&str; 4]> = smallvec!["abc"];
        assert_eq!(trans.fields["data"], expected);

        let expected_duration = ((now - ChronoDuration::seconds(10)).timestamp_micros()
            - (now - ChronoDuration::seconds(30)).timestamp_micros())
            as f64
            / 1_000_000.0;
        assert_eq!(trans.get_duration().unwrap(), expected_duration);
        assert!(!trans.get_is_closed());
        trans.set_is_closed();
        assert!(trans.get_is_closed());
    }

    #[test]
    fn test_transaction_to_record_batch() {
        let mut trans = Transaction::new(smallvec!["user_id".to_string()], TransactionType::Normal);
        let now = Utc::now();
        let event1_time = now - ChronoDuration::seconds(20);
        let event2_time = now - ChronoDuration::seconds(10);

        trans
            .add_event(&create_event(
                event2_time.timestamp_micros(),
                "second message",
                &[("user_id", "user_a"), ("status", "completed")],
            ))
            .unwrap();
        trans
            .add_event(&create_event(
                event1_time.timestamp_micros(),
                "first message",
                &[("user_id", "user_a"), ("source", "web")],
            ))
            .unwrap();
        trans.set_is_closed();

        let transactions = vec![trans];
        let rb = to_record_batch(&TransParams::default(), &transactions)
            .unwrap()
            .unwrap();

        // 8 columns: _time, _message, _duration, _event_count, _is_closed, source, status, user_id (sorted)
        assert_eq!(rb.num_columns(), 8);
        assert_eq!(rb.num_rows(), 1);

        let time_array = rb
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(time_array.value(0), event1_time.timestamp_micros());

        let message_values = extract_list_string_values(rb.column(1));
        assert_eq!(message_values, vec!["first message", "second message"]);

        let duration_array = rb
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(
            duration_array.value(0),
            transactions[0].get_duration().unwrap()
        );

        let event_count_array = rb.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(event_count_array.value(0), 2);

        let is_closed_array = rb
            .column(4)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(is_closed_array.value(0));

        // Check values of dynamic fields, they are sorted by name
        assert_eq!(rb.schema().field(5).name(), "source");
        assert_eq!(extract_list_string_values(rb.column(5)), vec!["web"]);

        assert_eq!(rb.schema().field(6).name(), "status");
        assert_eq!(extract_list_string_values(rb.column(6)), vec!["completed"]);

        assert_eq!(rb.schema().field(7).name(), "user_id");
        assert_eq!(extract_list_string_values(rb.column(7)), vec!["user_a"]);
    }

    #[test]
    fn test_trans_pool_add_events_basic_scenario() {
        let params = TransParams::new(
            Some(vec![]),
            vec![("fields".to_string(), Arg::String("user_id".to_string()))],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        let event1_time = now - ChronoDuration::seconds(5);
        let event2_time = now - ChronoDuration::seconds(10);
        let event3_time = now - ChronoDuration::seconds(20);
        let event4_time = now - ChronoDuration::seconds(30);

        // Event 1
        let event1 = create_event(
            event1_time.timestamp_micros(),
            "login",
            &[("user_id", "user1")],
        );
        pool.add_event(event1).unwrap();
        assert_eq!(pool.live_trans.len(), 1);
        assert_eq!(pool.frozen_trans.len(), 0);

        // Event 2 (same user)
        let event2 = create_event(
            event2_time.timestamp_micros(),
            "login",
            &[("user_id", "user1")],
        );

        pool.add_event(event2).unwrap();
        assert_eq!(pool.live_trans.len(), 1);
        assert_eq!(pool.frozen_trans.len(), 0);
        assert_eq!(
            pool.live_trans.values().next().unwrap().get_event_count(),
            2
        );

        // Event 3 (different user)
        let event3 = create_event(
            event3_time.timestamp_micros(),
            "login",
            &[("user_id", "user2")],
        );
        pool.add_event(event3).unwrap();
        assert_eq!(pool.live_trans.len(), 2);
        assert_eq!(pool.frozen_trans.len(), 0);

        let event4 = create_event(
            event4_time.timestamp_micros(),
            "logout",
            &[("user_id", "user1")],
        );
        pool.add_event(event4).unwrap();
        assert_eq!(pool.live_trans.len(), 2);
        assert_eq!(pool.frozen_trans.len(), 0);
        assert_eq!(
            pool.live_trans
                .get(&smallvec![Some("user1".to_string())])
                .unwrap()
                .get_event_count(),
            3
        );
        assert_eq!(
            pool.live_trans
                .get(&smallvec![Some("user2".to_string())])
                .unwrap()
                .get_event_count(),
            1
        );

        let mut pool_max_events = TransactionPool::new(
            TransParams::new(
                Some(vec![]),
                vec![
                    ("fields".to_string(), Arg::String("user_id".to_string())),
                    ("max_events".to_string(), Arg::Int(3)),
                ],
            )
            .unwrap(),
        );
        let now_fe = Utc::now();
        pool_max_events
            .add_event(create_event(
                now_fe.timestamp_micros(),
                "event A",
                &[("user_id", "userX")],
            ))
            .unwrap();
        pool_max_events
            .add_event(create_event(
                (now_fe - ChronoDuration::seconds(1)).timestamp_micros(),
                "event B",
                &[("user_id", "userX")],
            ))
            .unwrap();
        assert_eq!(pool_max_events.live_trans.len(), 1);
        assert_eq!(pool_max_events.frozen_trans.len(), 0);
        pool_max_events
            .add_event(create_event(
                (now_fe - ChronoDuration::seconds(2)).timestamp_micros(),
                "event C",
                &[("user_id", "userX")],
            ))
            .unwrap();
        assert_eq!(pool_max_events.live_trans.len(), 0); // userX trans should be frozen
        assert_eq!(pool_max_events.frozen_trans.len(), 1);
        assert_eq!(pool_max_events.frozen_trans[0].get_event_count(), 3);
    }

    #[test]
    fn test_trans_pool_max_span_restriction() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("id".to_string())),
                ("max_span".to_string(), Arg::String("10s".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        // Event 1 for trans_a (within span)
        let event1 = create_event(
            (now - ChronoDuration::seconds(15)).timestamp_micros(),
            "event_a_2",
            &[("id", "trans_a")],
        );
        // Event 2 for trans_a
        let event2 = create_event(
            (now - ChronoDuration::seconds(20)).timestamp_micros(),
            "event_a_1",
            &[("id", "trans_a")],
        );
        let event3 = create_event(
            (now - ChronoDuration::seconds(25)).timestamp_micros(),
            "event_b_2",
            &[("id", "trans_b")],
        );
        // Event 4 for trans_b
        let event4 = create_event(
            (now - ChronoDuration::seconds(30)).timestamp_micros(),
            "event_b_1",
            &[("id", "trans_b")],
        );
        let event5_time = now - ChronoDuration::seconds(45);
        let event5 = create_event(
            event5_time.timestamp_micros(),
            "event_c_1",
            &[("id", "trans_c")],
        );
        let event6_time = now - ChronoDuration::seconds(80);
        let event6 = create_event(
            event6_time.timestamp_micros(),
            "event_c_1",
            &[("id", "trans_c")],
        );

        pool.add_event(event1).unwrap();
        pool.add_event(event2).unwrap();
        pool.add_event(event3).unwrap();
        pool.add_event(event4).unwrap();
        pool.add_event(event5).unwrap();
        pool.add_event(event6).unwrap();

        assert_eq!(pool.live_trans.len(), 1);
        assert_eq!(pool.frozen_trans.len(), 3);
    }

    #[test]
    fn test_trans_pool_multi_value_fields() {
        let params = TransParams::new(
            Some(vec![]),
            vec![(
                "fields".to_string(),
                Arg::String("client_ip,status".to_string()),
            )],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        // Event 1
        let event1 = create_event(
            now.timestamp_micros(),
            "msg1",
            &[("client_ip", "1.1.1.1"), ("status", "200")],
        );
        pool.add_event(event1).unwrap();
        assert_eq!(pool.live_trans.len(), 1);
        let key1 = smallvec![Some("1.1.1.1".to_string()), Some("200".to_string())];
        assert!(pool.live_trans.contains_key(&key1));

        // Event 2 (same client_ip, different status) -> new transaction
        let event2 = create_event(
            (now - ChronoDuration::seconds(1)).timestamp_micros(),
            "msg2",
            &[("client_ip", "1.1.1.1"), ("status", "404")],
        );
        pool.add_event(event2).unwrap();
        assert_eq!(pool.live_trans.len(), 2); // Two live transactions
        let key2 = smallvec![Some("1.1.1.1".to_string()), Some("404".to_string())];
        assert!(pool.live_trans.contains_key(&key2));
        assert_eq!(pool.live_trans.get(&key1).unwrap().get_event_count(), 1);
        assert_eq!(pool.live_trans.get(&key2).unwrap().get_event_count(), 1);

        // Event 3 (different client_ip, same status as key1) -> new transaction
        let event3 = create_event(
            (now - ChronoDuration::seconds(2)).timestamp_micros(),
            "msg3",
            &[("client_ip", "2.2.2.2"), ("status", "200")],
        );
        pool.add_event(event3).unwrap();
        assert_eq!(pool.live_trans.len(), 3); // Three live transactions
        let key3 = smallvec![Some("2.2.2.2".to_string()), Some("200".to_string())];
        assert!(pool.live_trans.contains_key(&key3));

        // Event 4 (same as Event 1 key) -> merges with key1 transaction
        let event4 = create_event(
            (now - ChronoDuration::seconds(3)).timestamp_micros(),
            "msg4",
            &[("client_ip", "1.1.1.1"), ("status", "200")],
        );
        pool.add_event(event4).unwrap();
        assert_eq!(pool.live_trans.len(), 3); // Still three live transactions
        assert_eq!(pool.live_trans.get(&key1).unwrap().get_event_count(), 2);
    }

    #[test]
    fn test_trans_pool_starts_ends_if_field() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("session_id".to_string())),
                (
                    "starts_if_field".to_string(),
                    Arg::String("start_flag".to_string()),
                ),
                (
                    "ends_if_field".to_string(),
                    Arg::String("end_flag".to_string()),
                ),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        let event1 = create_event(
            now.timestamp_micros(),
            "event_A1",
            &[("session_id", "A"), ("start_flag", "true")],
        );
        let event2 = create_event(
            (now - ChronoDuration::seconds(1)).timestamp_micros(),
            "event_A2",
            &[("session_id", "A")],
        );
        let event3 = create_event(
            (now - ChronoDuration::seconds(2)).timestamp_micros(),
            "event_A3",
            &[("session_id", "A"), ("end_flag", "true")],
        );
        let event4 = create_event(
            (now - ChronoDuration::seconds(3)).timestamp_micros(),
            "event_B1",
            &[
                ("session_id", "B"),
                ("start_flag", "true"),
                ("end_flag", "true"),
            ],
        );
        let event5 = create_event(
            (now - ChronoDuration::seconds(4)).timestamp_micros(),
            "event_C1",
            &[("session_id", "C"), ("start_flag", "true")],
        );
        let event6 = create_event(
            (now - ChronoDuration::seconds(5)).timestamp_micros(),
            "event_C2",
            &[("session_id", "C")],
        );
        let event7 = create_event(
            (now - ChronoDuration::seconds(6)).timestamp_micros(),
            "event_C3",
            &[("session_id", "C"), ("start_flag", "true")],
        );

        pool.add_event(event1).unwrap();
        pool.add_event(event2).unwrap();
        pool.add_event(event3).unwrap();
        pool.add_event(event4).unwrap();
        pool.add_event(event5).unwrap();
        pool.add_event(event6).unwrap();
        pool.add_event(event7).unwrap();

        assert_eq!(pool.live_trans.len(), 1);
        assert_eq!(pool.frozen_trans.len(), 5);
        assert_eq!(
            pool.frozen_trans
                .last()
                .unwrap()
                .messages
                .iter()
                .flatten()
                .collect::<Vec<_>>(),
            vec!["event_C3", "event_C2"]
        );
    }

    #[test]
    fn test_trans_pool_ignore_nulls_in_key() {
        let params = TransParams::new(
            Some(vec![]),
            vec![(
                "fields".to_string(),
                Arg::String("session_id,host".to_string()),
            )],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        let events = vec![
            create_event_nullable(
                now.timestamp_micros(),
                "a",
                &[("session_id", Some("A")), ("host", None)],
            ),
            create_event(
                now.timestamp_micros(),
                "b",
                &[("session_id", "A"), ("host", "localhost")],
            ),
            create_event(
                now.timestamp_micros(),
                "c",
                &[("session_id", "B"), ("host", "localhost")],
            ),
            create_event(
                now.timestamp_micros(),
                "d",
                &[("session_id", "B"), ("host", "localhost")],
            ),
        ];

        for ev in events {
            pool.add_event(ev).unwrap();
        }

        let mut events = pool
            .live_trans
            .iter()
            .flat_map(|i| i.1.messages.iter().flatten())
            .collect::<Vec<_>>();
        events.sort_unstable();
        // With the new behavior, event "a" should also be included
        assert_eq!(events, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_trans_pool_bracket_matching() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("session_id".to_string())),
                (
                    "starts_with".to_string(),
                    Arg::String("starts_with".to_string()),
                ),
                (
                    "ends_with".to_string(),
                    Arg::String("ends_with".to_string()),
                ),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        // Create events in the order described in the issue:
        // 1. xxxxx ends_with xxxxx (session A)
        // 2. yyyyy ends_with yyyyy (session A)
        // 3. zzzzz starts_with zzzzz (session A)
        // 4. wwwww starts_with wwwww (session A)

        let event1 = create_event(
            now.timestamp_micros(),
            "xxxxx ends_with xxxxx",
            &[("session_id", "A")],
        );
        let event2 = create_event(
            (now - ChronoDuration::seconds(1)).timestamp_micros(),
            "yyyyy ends_with yyyyy",
            &[("session_id", "A")],
        );
        let event3 = create_event(
            (now - ChronoDuration::seconds(2)).timestamp_micros(),
            "zzzzz starts_with zzzzz",
            &[("session_id", "A")],
        );
        let event4 = create_event(
            (now - ChronoDuration::seconds(3)).timestamp_micros(),
            "wwwww starts_with wwwww",
            &[("session_id", "A")],
        );

        pool.add_event(event1).unwrap();
        pool.add_event(event2).unwrap();
        pool.add_event(event3).unwrap();
        pool.add_event(event4).unwrap();

        // With bracket matching logic:
        // - event3 (starts_with) should match with event2 (ends_with) -> transaction 2-3
        // - event4 (starts_with) should match with event1 (ends_with) -> transaction 1-4
        // This should result in 2 closed transactions

        // Apply post-processing
        pool.post_process_transactions();

        // Now we should have 2 transactions
        assert_eq!(pool.frozen_trans.len(), 2);

        // Check that both transactions are closed
        assert!(pool.frozen_trans.iter().all(|t| t.get_is_closed()));

        // Check that we have pending transactions for the unmatched starts_with events
        assert_eq!(pool.pending_transactions.len(), 0); // All should be matched
    }

    #[test]
    fn test_trans_pool_bracket_matching_with_field_conditions() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("session_id".to_string())),
                (
                    "starts_if_field".to_string(),
                    Arg::String("is_start".to_string()),
                ),
                (
                    "ends_if_field".to_string(),
                    Arg::String("is_end".to_string()),
                ),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        // Create events in the order described in the issue:
        // 1. message1 (is_end=true) (session A)
        // 2. message2 (is_end=true) (session A)
        // 3. message3 (is_start=true) (session A)
        // 4. message4 (is_start=true) (session A)

        let event1 = create_event(
            now.timestamp_micros(),
            "message1",
            &[("session_id", "A"), ("is_end", "true")],
        );
        let event2 = create_event(
            (now - ChronoDuration::seconds(1)).timestamp_micros(),
            "message2",
            &[("session_id", "A"), ("is_end", "true")],
        );
        let event3 = create_event(
            (now - ChronoDuration::seconds(2)).timestamp_micros(),
            "message3",
            &[("session_id", "A"), ("is_start", "true")],
        );
        let event4 = create_event(
            (now - ChronoDuration::seconds(3)).timestamp_micros(),
            "message4",
            &[("session_id", "A"), ("is_start", "true")],
        );

        pool.add_event(event1).unwrap();
        pool.add_event(event2).unwrap();
        pool.add_event(event3).unwrap();
        pool.add_event(event4).unwrap();

        // With bracket matching logic:
        // - event3 (is_start=true) should match with event2 (is_end=true) -> transaction 2-3
        // - event4 (is_start=true) should match with event1 (is_end=true) -> transaction 1-4
        // This should result in 2 closed transactions

        // Apply post-processing
        pool.post_process_transactions();

        // Now we should have 2 transactions
        assert_eq!(pool.frozen_trans.len(), 2);

        // Check that both transactions are closed
        assert!(pool.frozen_trans.iter().all(|t| t.get_is_closed()));
    }

    #[test]
    fn test_trans_pool_discard_all_nulls() {
        let params = TransParams::new(
            Some(vec![]),
            vec![(
                "fields".to_string(),
                Arg::String("session_id,host".to_string()),
            )],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        let events = vec![
            // Event with all transkey fields null should be discarded
            create_event_nullable(
                now.timestamp_micros(),
                "discarded",
                &[("session_id", None), ("host", None)],
            ),
            // Event with some values should be included
            create_event(
                now.timestamp_micros(),
                "included",
                &[("session_id", "A"), ("host", "localhost")],
            ),
        ];

        for ev in events {
            pool.add_event(ev).unwrap();
        }

        // Get all messages from all transactions
        let mut all_messages: Vec<String> = Vec::new();
        for (_, trans) in &pool.live_trans {
            for messages in &trans.messages {
                for message in messages {
                    all_messages.push(message.clone());
                }
            }
        }

        // Also get messages from frozen transactions
        for trans in &pool.frozen_trans {
            for messages in &trans.messages {
                for message in messages {
                    all_messages.push(message.clone());
                }
            }
        }

        // Only the "included" event should be present
        assert_eq!(all_messages, vec!["included"]);

        // Check that we have only 1 transaction (the "discarded" event should be ignored)
        assert_eq!(pool.live_trans.len() + pool.frozen_trans.len(), 1);
    }

    #[test]
    fn test_trans_pool_example_scenario() {
        let params = TransParams::new(
            Some(vec![]),
            vec![("fields".to_string(), Arg::String("key,key2".to_string()))],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let base_time = 1750844229000000i64;

        // Create events as per the example:
        // (1750844229000000, 'xxx', 'kkk', '1') -> should be one event
        // (1750844228000000, 'xxx', 'jjj', '2') -> should be another event
        // (1750844227000000, 'yyy', 'lll', '3') -> should be part of a transaction with the next two
        // (1750844226000000, null, 'lll', '4') -> should be part of the same transaction as above
        // (1750844225000000, null, 'lll', '5') -> should be part of the same transaction as above

        let events = vec![
            create_event(base_time, "1", &[("key", "xxx"), ("key2", "kkk")]),
            create_event(base_time - 1000000, "2", &[("key", "xxx"), ("key2", "jjj")]),
            create_event(base_time - 2000000, "3", &[("key", "yyy"), ("key2", "lll")]),
            create_event_nullable(
                base_time - 3000000,
                "4",
                &[("key", None), ("key2", Some("lll"))],
            ),
            create_event_nullable(
                base_time - 4000000,
                "5",
                &[("key", None), ("key2", Some("lll"))],
            ),
        ];

        for ev in events {
            pool.add_event(ev).unwrap();
        }

        // Get all messages from all transactions
        let mut all_messages: Vec<String> = Vec::new();
        for (_, trans) in &pool.live_trans {
            for messages in &trans.messages {
                for message in messages {
                    all_messages.push(message.clone());
                }
            }
        }

        // Also get messages from frozen transactions
        for trans in &pool.frozen_trans {
            for messages in &trans.messages {
                for message in messages {
                    all_messages.push(message.clone());
                }
            }
        }

        // All messages should be present
        all_messages.sort_unstable();
        assert_eq!(all_messages, vec!["1", "2", "3", "4", "5"]);

        // Check that we have 3 transactions:
        // 1. Event "1" (key=xxx, key2=kkk)
        // 2. Event "2" (key=xxx, key2=jjj)
        // 3. Events "3", "4", "5" (key2=lll) - should be grouped together because key2 matches
        assert_eq!(pool.live_trans.len() + pool.frozen_trans.len(), 3);
    }
}
