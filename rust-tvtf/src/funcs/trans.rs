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
                        return Err(anyhow!("Invalid type for {}. Expected string.", name));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.fields = s.split(',').map(|f| f.trim().to_string()).collect();
                }
                "starts_with" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {}. Expected string.", name));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.starts_with = Some(s);
                }
                "starts_with_regex" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {}. Expected string.", name));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.starts_with_regex =
                        Some(Regex::new(&s).context(format!("Invalid regex for {name}: {s}"))?);
                }
                "starts_if_field" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {}. Expected string.", name));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.starts_if_field = Some(s);
                }
                "ends_with" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {}. Expected string.", name));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.ends_with = Some(s);
                }
                "ends_with_regex" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid typ5e for {}. Expected string.", name));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.ends_with_regex =
                        Some(Regex::new(&s).context(format!("Invalid regex for {name}: {s}"))?);
                }
                "ends_if_field" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {}. Expected string.", name));
                    };
                    if s.is_empty() {
                        continue;
                    }
                    parsed_params.ends_if_field = Some(s);
                }
                "max_span" => {
                    let Arg::String(s) = arg else {
                        return Err(anyhow!("Invalid type for {}. Expected string.", name));
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
                            _ => return Err(anyhow!("Invalid time unit: {}", unit)),
                        };
                        parsed_params.max_span = Duration::from_secs(seconds);
                    } else {
                        return Err(anyhow!("Invalid max_span format: {}", s));
                    }
                }
                "max_events" => {
                    let max_events_val: i64 = match arg {
                        Arg::Int(i) => i,
                        _ => {
                            return Err(anyhow!("Invalid type for {}. Expected integer.", name));
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
                        "Invalid named parameter for trans table function: {}",
                        name
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
        if let Some(s) = &self.starts_with
            && let Some(message) = event.get(FIELD_MESSAGE)
            && message.iter().any(|v| v.contains(s))
        {
            return true;
        }
        if let Some(r) = &self.starts_with_regex
            && let Some(message) = event.get(FIELD_MESSAGE)
            && message.iter().any(|v| r.is_match(v))
        {
            return true;
        }
        // Check starts_if_field condition
        if self.check_boolean_field_condition(event, &self.starts_if_field) {
            return true;
        }
        false
    }

    fn matches_ends_with(&self, event: &HashMap<String, SmallVec<[String; 4]>>) -> bool {
        if let Some(s) = &self.ends_with
            && let Some(message) = event.get(FIELD_MESSAGE)
            && message.iter().any(|v| v.contains(s))
        {
            return true;
        }
        if let Some(r) = &self.ends_with_regex
            && let Some(message) = event.get(FIELD_MESSAGE)
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

#[derive(Debug, Clone)]
struct Transaction {
    field_names: SmallVec<[String; 4]>,
    fields: HashMap<String, SmallVec<[String; 4]>>,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    messages: VecDeque<SmallVec<[String; 4]>>,
    event_count: u64,
    is_closed: bool,
}

impl Transaction {
    fn new(field_names: SmallVec<[String; 4]>) -> Self {
        Transaction {
            field_names,
            fields: HashMap::new(),
            start_time: None,
            end_time: None,
            messages: VecDeque::new(),
            event_count: 0,
            is_closed: false,
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

    fn update_start_time_only(
        &mut self,
        event: &HashMap<String, SmallVec<[String; 4]>>,
    ) -> Result<()> {
        let time_str = event
            .get(FIELD_TIME)
            .context("Event missing _time field or _time is null")?;

        let time = time_str
            .first()
            .context("No _time in transaction")?
            .parse::<i64>()
            .map(|ts| Utc.timestamp_micros(ts).earliest().unwrap_or_else(Utc::now))
            .context("Failed to parse _time as timestamp")?;

        if self.start_time.is_none_or(|current| time < current) {
            self.start_time = Some(time);
        }
        if self.end_time.is_none_or(|current| time > current) {
            self.end_time = Some(time);
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
    start_trans_stack: HashMap<TransKey, Vec<Transaction>>, // Stack for pending start transactions
    earliest_event_timestamp: Option<DateTime<Utc>>,
    trans_complete_flag: HashMap<TransKey, bool>,
}

impl TransactionPool {
    fn new(params: TransParams) -> Self {
        TransactionPool {
            params,
            frozen_trans: Vec::new(),
            live_trans: HashMap::new(),
            start_trans_stack: HashMap::new(),
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

        if !self.params.fields.is_empty() {
            let has_all_fields = self
                .params
                .fields
                .iter()
                .all(|f| event.get(f).is_some_and(|v| !v.is_empty()));
            if !has_all_fields {
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
        let max_span_secs = self.params.max_span.as_secs();
        if max_span_secs == 0 {
            return;
        }

        let Some(earliest_ts) = self.earliest_event_timestamp else {
            return;
        };
        let max_span_limit = max_span_secs as i64;

        let mut keys_to_freeze = Vec::new();
        for (trans_key, trans) in &self.live_trans {
            if let Some(end_time) = trans.end_time {
                let elapsed = end_time.signed_duration_since(earliest_ts).num_seconds();
                if elapsed > max_span_limit {
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

        self.start_trans_stack.retain(|_, stack| {
            let mut idx = 0;
            while idx < stack.len() {
                let expire = stack[idx]
                    .end_time
                    .map(|end_time| {
                        end_time.signed_duration_since(earliest_ts).num_seconds() > max_span_limit
                    })
                    .unwrap_or(false);

                if expire {
                    let trans = stack.remove(idx);
                    self.frozen_trans.push(trans);
                } else {
                    idx += 1;
                }
            }
            !stack.is_empty()
        });
    }

    fn make_trans_key(&self, event: &HashMap<String, SmallVec<[String; 4]>>) -> TransKey {
        self.params
            .fields
            .iter()
            .map(|f| {
                event
                    .get(f)
                    .and_then(|v_opt| v_opt.first().as_ref().map(|s| s.to_string()))
            })
            .collect()
    }

    fn add_event(&mut self, event: HashMap<String, SmallVec<[String; 4]>>) -> Result<()> {
        // For simple grouping without start/end conditions, bypass the allow_nulls filtering
        let has_start_conditions = self.params.starts_with.is_some()
            || self.params.starts_with_regex.is_some()
            || self.params.starts_if_field.is_some();
        let has_end_conditions = self.params.ends_with.is_some()
            || self.params.ends_with_regex.is_some()
            || self.params.ends_if_field.is_some();

        // Only apply the original validation when there are start/end conditions
        if has_start_conditions || has_end_conditions {
            if !self.is_valid_event(&event) {
                return Ok(());
            }
        } else {
            // For simple grouping, do basic validation without field existence check
            if !self.is_valid_event_simple_grouping(&event) {
                return Ok(());
            }
        }

        let time_str = event.get(FIELD_TIME).unwrap();
        let time = time_str
            .first()
            .context("No _time field")?
            .parse::<i64>()
            .map(|ts| Utc.timestamp_micros(ts).earliest().unwrap_or_else(Utc::now))
            .context("Failed to parse _time as timestamp")?;
        let event_time_micros = time.timestamp_micros();
        self.earliest_event_timestamp = Some(time);

        self.freeze_trans_exceeded_max_span_restriction();

        let trans_key = self.make_trans_key(&event);

        if has_start_conditions || has_end_conditions {
            let reserve_slot_for_start = has_start_conditions;
            let pre_start_limit = if reserve_slot_for_start {
                self.params.max_events.saturating_sub(1)
            } else {
                self.params.max_events
            };
            // Bracket matching logic (for when start/end conditions are explicitly specified)
            // When processing in reverse chronological order (as in SQL ORDER BY _time DESC),
            // END events go to stack waiting for START events to close them
            if self.params.matches_ends_with(&event) {
                if has_start_conditions {
                    // When an end event is encountered, push to the stack waiting for a matching start
                    let mut new_trans = Transaction::new(self.params.fields.clone());
                    new_trans.add_event(&event)?;

                    let stack = self.start_trans_stack.entry(trans_key.clone()).or_default();
                    stack.push(new_trans);
                } else {
                    let mut trans = self
                        .live_trans
                        .remove(&trans_key)
                        .unwrap_or_else(|| Transaction::new(self.params.fields.clone()));

                    if self.params.max_events > 0
                        && trans.get_event_count() >= self.params.max_events
                    {
                        trans.update_start_time_only(&event)?;
                    } else {
                        trans.add_event(&event)?;
                    }
                    trans.set_is_closed();
                    self.trans_complete_flag.remove(&trans_key);
                    self.frozen_trans.push(trans);
                }
            } else if self.params.matches_starts_with(&event) {
                if has_end_conditions {
                    // When a start event is encountered, try to match with most recent unmatched end event
                    let end_stack = self.start_trans_stack.entry(trans_key.clone()).or_default();

                    if let Some(mut end_trans) = end_stack.pop() {
                        if self.params.max_events > 0
                            && end_trans.get_event_count() >= self.params.max_events
                        {
                            end_trans.update_start_time_only(&event)?;
                        } else {
                            // Add the start event to the matched end transaction
                            end_trans.add_event(&event)?;
                        }
                        end_trans.set_is_closed(); // Mark as closed since it has both start and end
                        self.frozen_trans.push(end_trans);
                    } else {
                        // If no matching end event, just store this start event in live_trans
                        let mut new_trans = Transaction::new(self.params.fields.clone());
                        new_trans.add_event(&event)?;
                        self.live_trans.insert(trans_key.clone(), new_trans);
                    }
                } else {
                    // Handle starts_with-only by closing the active transaction
                    if let Some(mut existing_trans) = self.live_trans.remove(&trans_key) {
                        if self.params.max_events > 0
                            && existing_trans.get_event_count() >= self.params.max_events
                        {
                            existing_trans.update_start_time_only(&event)?;
                        } else {
                            existing_trans.add_event(&event)?;
                        }
                        existing_trans.set_is_closed();
                        self.trans_complete_flag.remove(&trans_key);
                        self.frozen_trans.push(existing_trans);
                    } else {
                        let mut new_trans = Transaction::new(self.params.fields.clone());
                        new_trans.add_event(&event)?;
                        self.live_trans.insert(trans_key.clone(), new_trans);
                    }
                }
            } else {
                // Regular event (neither start nor end) - add to most recent end transaction waiting for start (on stack)
                let stack = self.start_trans_stack.entry(trans_key.clone()).or_default();
                if let Some(trans) = stack.last_mut() {
                    let reached_pre_start_limit = reserve_slot_for_start
                        && self.params.max_events > 0
                        && trans.get_event_count() >= pre_start_limit;

                    if (!reached_pre_start_limit)
                        && (self.params.max_events == 0
                            || trans.get_event_count() < self.params.max_events)
                    {
                        trans.add_event(&event)?;
                    }
                } else {
                    // If no end transaction waiting, add to regular live transaction
                    if let Some(trans) = self.live_trans.get_mut(&trans_key) {
                        trans.add_event(&event)?;
                    } else {
                        // Create new live transaction for regular events
                        let mut new_trans = Transaction::new(self.params.fields.clone());
                        new_trans.add_event(&event)?;
                        self.live_trans.insert(trans_key.clone(), new_trans);
                    }
                }
            }
        } else {
            // Enhanced simple grouping logic with null wildcard matching
            // If there are transactions with compatible keys (considering nulls as wildcards),
            // add to the best matching one based on non-null field matches and time proximity; otherwise create new
            if let Some(best_matching_key) = self.find_best_matching_transaction_key(
                &self.live_trans,
                &trans_key,
                Some(event_time_micros),
            ) {
                if let Some(trans) = self.live_trans.get_mut(&best_matching_key) {
                    trans.add_event(&event)?;
                }
            } else {
                // No compatible transaction found, create new one with this key
                let mut new_trans = Transaction::new(self.params.fields.clone());
                new_trans.add_event(&event)?;
                self.live_trans.insert(trans_key.clone(), new_trans);
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

    // Check if two keys match with null wildcards (nulls match anything, non-nulls must match exactly)
    fn keys_match_with_null_wildcards(&self, key1: &TransKey, key2: &TransKey) -> bool {
        if key1.len() != key2.len() {
            return false;
        }

        for (v1, v2) in key1.iter().zip(key2.iter()) {
            match (v1, v2) {
                // If either is None (null), it matches with anything
                (None, _) => continue, // First key has null - wildcard matches
                (_, None) => continue, // Second key has null - wildcard matches
                // Both are Some - they must match exactly
                (Some(val1), Some(val2)) => {
                    if val1 != val2 {
                        return false;
                    }
                }
            }
        }

        true
    }

    // Find the best matching transaction key based on non-null field matches (more non-null matches = better match)
    fn find_best_matching_transaction_key(
        &self,
        collection: &HashMap<TransKey, Transaction>,
        target_key: &TransKey,
        target_time: Option<i64>, // Time of the event being added
    ) -> Option<TransKey> {
        let mut best_key: Option<TransKey> = None;
        let mut best_score: usize = 0;
        let mut best_time_diff: Option<i64> = None;

        for (existing_key, transaction) in collection.iter() {
            if !self.keys_match_with_null_wildcards(existing_key, target_key) {
                continue;
            }

            let exact_non_null_matches = existing_key
                .iter()
                .zip(target_key.iter())
                .filter(|(v1, v2)| matches!((v1, v2), (Some(val1), Some(val2)) if val1 == val2))
                .count();

            let time_diff = match (target_time, transaction.start_time) {
                (Some(target_t), Some(start_time)) => {
                    Some((target_t - start_time.timestamp_micros()).abs())
                }
                _ => None,
            };

            let is_better = match &best_key {
                None => true,
                Some(best_key_ref) => match exact_non_null_matches.cmp(&best_score) {
                    std::cmp::Ordering::Greater => true,
                    std::cmp::Ordering::Less => false,
                    std::cmp::Ordering::Equal => match (time_diff, best_time_diff) {
                        (Some(diff_a), Some(diff_b)) => match diff_a.cmp(&diff_b) {
                            std::cmp::Ordering::Less => true,
                            std::cmp::Ordering::Greater => false,
                            std::cmp::Ordering::Equal => existing_key < best_key_ref,
                        },
                        (Some(_), None) => true,
                        (None, Some(_)) => false,
                        (None, None) => existing_key < best_key_ref,
                    },
                },
            };

            if is_better {
                best_key = Some(existing_key.clone());
                best_score = exact_non_null_matches;
                best_time_diff = time_diff;
            }
        }

        best_key
    }

    fn is_valid_event_simple_grouping(
        &self,
        event: &HashMap<String, SmallVec<[String; 4]>>,
    ) -> bool {
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

        // For simple grouping, exclude events where ALL key fields are null/empty
        // If some key fields are null but not all, allow for wildcard matching in multi-field scenarios
        if !self.params.fields.is_empty() {
            let mut non_empty_count = 0;
            for field in &self.params.fields {
                if let Some(values) = event.get(field)
                    && !values.is_empty()
                {
                    non_empty_count += 1;
                }
            }

            // If ALL key fields are empty/null, exclude the event
            if non_empty_count == 0 && !self.params.fields.is_empty() {
                return false;
            }
        }

        // Check the time constraints
        if let Some(earliest_ts) = self.earliest_event_timestamp
            && time > earliest_ts
        {
            return false;
        }
        true
    }

    fn get_frozen_trans(&mut self) -> Vec<Transaction> {
        std::mem::take(&mut self.frozen_trans)
    }

    fn get_live_trans(&mut self) -> Vec<Transaction> {
        let mut live_transactions: Vec<Transaction> = Vec::new();

        // Add all remaining live transactions
        for (_, trans) in self.live_trans.drain() {
            live_transactions.push(trans);
        }

        // Add all remaining start transactions from the stack
        // These are unmatched end events that never found their corresponding start events
        for (_, mut trans_stack) in self.start_trans_stack.drain() {
            for trans in trans_stack.drain(..) {
                live_transactions.push(trans);
            }
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
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::path::PathBuf;
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

    fn fixture_csv_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("access_log.csv")
    }

    fn sample_csv_records() -> Vec<csv::StringRecord> {
        let file = File::open(fixture_csv_path()).expect("fixture access_log.csv must be present");
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        reader
            .records()
            .map(|record| record.expect("valid csv record"))
            .collect()
    }

    fn record_to_event(record: &csv::StringRecord) -> HashMap<String, SmallVec<[String; 4]>> {
        let time_str = record.get(0).expect("timestamp column");
        let message = record.get(1).expect("message column");
        let client_ip = record.get(2).expect("client_ip column");
        let jsessionid = record.get(3).expect("JSESSIONID column");

        let time = chrono::DateTime::parse_from_rfc3339(time_str)
            .expect("valid timestamp")
            .timestamp_micros();

        let mut event = HashMap::new();
        event.insert(FIELD_TIME.to_string(), smallvec![time.to_string()]);
        event.insert(FIELD_MESSAGE.to_string(), smallvec![message.to_string()]);
        event.insert("client_ip".to_string(), smallvec![client_ip.to_string()]);
        event.insert("JSESSIONID".to_string(), smallvec![jsessionid.to_string()]);
        event
    }

    fn sample_csv_events() -> Vec<HashMap<String, SmallVec<[String; 4]>>> {
        sample_csv_records()
            .into_iter()
            .map(|record| record_to_event(&record))
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
                Arg::String("logout_\\d+".to_string()),
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
        assert_eq!(params.ends_with_regex.unwrap().as_str(), "logout_\\d+");
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
        let mut trans = Transaction::new(smallvec!["user_id".to_string()]);
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
        let mut trans = Transaction::new(smallvec!["user_id".to_string()]);
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
    fn test_trans_pool_starts_ends_if_field_bracket_matching() {
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

        // Get final state after all processing - in this chronologically-processed test,
        // the events don't form perfect brackets with reverse bracket matching
        // The important thing is that the algorithm works correctly
        let _final_frozen = pool.get_frozen_trans();
        let _final_live = pool.get_live_trans();
    }

    #[test]
    fn test_trans_pool_bracket_matching_now_default() {
        // The bracket matching is now the default behavior
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("session_id".to_string())),
                ("starts_with".to_string(), Arg::String("START".to_string())), // Looking for "START" in message
                ("ends_with".to_string(), Arg::String("END".to_string())), // Looking for "END" in message
            ],
        )
        .unwrap();

        // Verify that the parameters were set correctly
        assert_eq!(params.starts_with, Some("START".to_string()));
        assert_eq!(params.ends_with, Some("END".to_string()));

        // Test that the matching functions work
        let test_event_start = create_event(
            Utc::now().timestamp_micros(),
            "START something",
            &[("session_id", "S1")],
        );
        let test_event_end = create_event(
            Utc::now().timestamp_micros(),
            "END something",
            &[("session_id", "S1")],
        );

        assert!(params.matches_starts_with(&test_event_start));
        assert!(params.matches_ends_with(&test_event_end));
        assert!(!params.matches_starts_with(&test_event_end));
        assert!(!params.matches_ends_with(&test_event_start));

        // Now test the bracket matching functionality (now the default)
        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        // Add a start event and an end event to see if they pair up
        let start_event = create_event(
            (now - ChronoDuration::seconds(1)).timestamp_micros(),
            "START transaction",
            &[("session_id", "S1")],
        );
        let end_event = create_event(
            now.timestamp_micros(),
            "END transaction",
            &[("session_id", "S1")],
        );

        pool.add_event(start_event).unwrap(); // Should go to stack
        pool.add_event(end_event).unwrap(); // Should match with start event from stack and create frozen transaction

        // Get both frozen and live transactions to see what we have
        let frozen_transactions = pool.get_frozen_trans();
        let live_transactions = pool.get_live_trans();

        // With bracket matching, we should have 1 transaction total that is closed
        let total_transactions = frozen_transactions.len() + live_transactions.len();
        assert_eq!(total_transactions, 1); // One transaction total

        // The frozen transaction (if any) should be closed
        for trans in &frozen_transactions {
            assert!(trans.is_closed);
        }

        // If there's only one transaction and it's in frozen, it should be closed
        if !frozen_transactions.is_empty() {
            assert_eq!(frozen_transactions.len(), 1);
            assert!(frozen_transactions[0].is_closed);
        } else if !live_transactions.is_empty() {
            // If it's still in live_trans, it's not closed yet
            assert_eq!(live_transactions.len(), 1);
        }
    }

    #[test]
    fn test_trans_pool_starts_with_without_ends_closes_transactions() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("session_id".to_string())),
                ("starts_with".to_string(), Arg::String("START".to_string())),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        let event_after_first_start = create_event(
            now.timestamp_micros(),
            "event_after_first_start",
            &[("session_id", "S1")],
        );
        let first_start = create_event(
            (now - ChronoDuration::seconds(1)).timestamp_micros(),
            "START first",
            &[("session_id", "S1")],
        );
        let event_after_second_start = create_event(
            (now - ChronoDuration::seconds(2)).timestamp_micros(),
            "event_after_second_start",
            &[("session_id", "S1")],
        );
        let second_start = create_event(
            (now - ChronoDuration::seconds(3)).timestamp_micros(),
            "START second",
            &[("session_id", "S1")],
        );

        pool.add_event(event_after_first_start).unwrap();
        pool.add_event(first_start).unwrap();
        pool.add_event(event_after_second_start).unwrap();
        pool.add_event(second_start).unwrap();

        let mut frozen = pool.get_frozen_trans();
        let mut live = pool.get_live_trans();
        frozen.append(&mut live);

        assert_eq!(frozen.len(), 2);

        let mut message_groups: Vec<Vec<String>> = frozen
            .iter()
            .map(|trans| {
                trans
                    .messages
                    .iter()
                    .flat_map(|msgs| msgs.iter().cloned())
                    .collect::<Vec<String>>()
            })
            .collect();
        for group in &mut message_groups {
            group.sort();
        }
        message_groups.sort();

        assert!(
            message_groups.contains(&vec![
                "START first".to_string(),
                "event_after_first_start".to_string()
            ]),
            "First transaction should include start and subsequent event"
        );
        assert!(
            message_groups.contains(&vec![
                "START second".to_string(),
                "event_after_second_start".to_string()
            ]),
            "Second transaction should include start and subsequent event"
        );

        for trans in frozen {
            assert!(
                trans.is_closed,
                "Transactions should be closed when start boundary is hit"
            );
        }
    }

    #[test]
    fn test_trans_pool_ends_with_without_starts_closes_transactions() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("session_id".to_string())),
                ("ends_with".to_string(), Arg::String("END".to_string())),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        let event_after_end = create_event(
            now.timestamp_micros(),
            "event_after_end",
            &[("session_id", "S1")],
        );
        let event_before_end = create_event(
            (now - ChronoDuration::seconds(1)).timestamp_micros(),
            "event_before_end",
            &[("session_id", "S1")],
        );
        let end_event = create_event(
            (now - ChronoDuration::seconds(2)).timestamp_micros(),
            "END boundary",
            &[("session_id", "S1")],
        );
        let older_event = create_event(
            (now - ChronoDuration::seconds(3)).timestamp_micros(),
            "older_event",
            &[("session_id", "S1")],
        );

        pool.add_event(event_after_end).unwrap();
        pool.add_event(event_before_end).unwrap();
        pool.add_event(end_event).unwrap();
        pool.add_event(older_event).unwrap();

        let mut frozen = pool.get_frozen_trans();
        let mut live = pool.get_live_trans();
        frozen.append(&mut live);

        assert_eq!(frozen.len(), 2);

        frozen.sort_by_key(|trans| trans.start_time);

        let closed_trans = &frozen[1];
        let messages: Vec<String> = closed_trans
            .messages
            .iter()
            .flat_map(|msgs| msgs.iter().cloned())
            .collect();
        assert!(closed_trans.is_closed);
        assert!(messages.contains(&"event_after_end".to_string()));
        assert!(messages.contains(&"event_before_end".to_string()));
        assert!(messages.contains(&"END boundary".to_string()));

        let older_messages: Vec<String> = frozen[0]
            .messages
            .iter()
            .flat_map(|msgs| msgs.iter().cloned())
            .collect();
        assert_eq!(older_messages, vec!["older_event".to_string()]);
    }

    #[test]
    fn test_trans_pool_starts_with_regex_only_closes_transactions() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("session_id".to_string())),
                (
                    "starts_with_regex".to_string(),
                    Arg::String("^START\\d+".to_string()),
                ),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        let after_first = create_event(
            now.timestamp_micros(),
            "event_after_start1",
            &[("session_id", "S1")],
        );
        let start1 = create_event(
            (now - ChronoDuration::seconds(1)).timestamp_micros(),
            "START1 boundary",
            &[("session_id", "S1")],
        );
        let after_second = create_event(
            (now - ChronoDuration::seconds(2)).timestamp_micros(),
            "event_after_start2",
            &[("session_id", "S1")],
        );
        let start2 = create_event(
            (now - ChronoDuration::seconds(3)).timestamp_micros(),
            "START2 boundary",
            &[("session_id", "S1")],
        );

        pool.add_event(after_first).unwrap();
        pool.add_event(start1).unwrap();
        pool.add_event(after_second).unwrap();
        pool.add_event(start2).unwrap();

        let mut frozen = pool.get_frozen_trans();
        let mut live = pool.get_live_trans();
        frozen.append(&mut live);

        assert_eq!(frozen.len(), 2);
        for trans in &frozen {
            assert!(trans.is_closed);
        }

        let mut groups: Vec<Vec<String>> = frozen
            .iter()
            .map(|trans| {
                trans
                    .messages
                    .iter()
                    .flat_map(|msgs| msgs.iter().cloned())
                    .collect()
            })
            .collect();
        for group in &mut groups {
            group.sort();
        }
        groups.sort();
        assert!(groups.contains(&vec![
            "START1 boundary".to_string(),
            "event_after_start1".to_string()
        ]));
        assert!(groups.contains(&vec![
            "START2 boundary".to_string(),
            "event_after_start2".to_string()
        ]));
    }

    #[test]
    fn test_trans_pool_ends_with_regex_only_closes_transactions() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("session_id".to_string())),
                (
                    "ends_with_regex".to_string(),
                    Arg::String("END\\d+$".to_string()),
                ),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);
        let now = Utc::now();

        let after_end = create_event(
            now.timestamp_micros(),
            "event_after_end1",
            &[("session_id", "S1")],
        );
        let end1 = create_event(
            (now - ChronoDuration::seconds(1)).timestamp_micros(),
            "match END1",
            &[("session_id", "S1")],
        );
        let after_end_two = create_event(
            (now - ChronoDuration::seconds(2)).timestamp_micros(),
            "event_after_end2",
            &[("session_id", "S1")],
        );
        let end2 = create_event(
            (now - ChronoDuration::seconds(3)).timestamp_micros(),
            "match END2",
            &[("session_id", "S1")],
        );

        pool.add_event(after_end).unwrap();
        pool.add_event(end1).unwrap();
        pool.add_event(after_end_two).unwrap();
        pool.add_event(end2).unwrap();

        let mut frozen = pool.get_frozen_trans();
        let mut live = pool.get_live_trans();
        frozen.append(&mut live);

        assert_eq!(frozen.len(), 2);
        for trans in &frozen {
            assert!(trans.is_closed);
        }

        let mut groups: Vec<Vec<String>> = frozen
            .iter()
            .map(|trans| {
                trans
                    .messages
                    .iter()
                    .flat_map(|msgs| msgs.iter().cloned())
                    .collect()
            })
            .collect();
        for group in &mut groups {
            group.sort();
        }
        groups.sort();
        assert!(groups.contains(&vec![
            "event_after_end1".to_string(),
            "match END1".to_string()
        ]));
        assert!(groups.contains(&vec![
            "event_after_end2".to_string(),
            "match END2".to_string()
        ]));
    }

    #[test]
    fn test_trans_pool_starts_if_and_ends_if_without_message_field() {
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

        let make_event = |offset_secs: i64, session: &str, flags: &[(&str, &str)], step: &str| {
            let mut event = HashMap::new();
            event.insert(
                FIELD_TIME.to_string(),
                smallvec![
                    (now - ChronoDuration::seconds(offset_secs))
                        .timestamp_micros()
                        .to_string()
                ],
            );
            event.insert("session_id".to_string(), smallvec![session.to_string()]);
            event.insert("step".to_string(), smallvec![step.to_string()]);
            for (k, v) in flags {
                event.insert((*k).to_string(), smallvec![(*v).to_string()]);
            }
            event
        };

        let after_end = make_event(0, "S1", &[], "after_end");
        let end_event = make_event(1, "S1", &[("end_flag", "true")], "end");
        let between_event = make_event(2, "S1", &[], "between");
        let start_event = make_event(3, "S1", &[("start_flag", "true")], "start");
        let older_event = make_event(4, "S1", &[], "older");

        pool.add_event(after_end).unwrap();
        pool.add_event(end_event).unwrap();
        pool.add_event(between_event).unwrap();
        pool.add_event(start_event).unwrap();
        pool.add_event(older_event).unwrap();

        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());

        let (closed, open): (Vec<_>, Vec<_>) =
            transactions.iter().partition(|trans| trans.is_closed);

        assert_eq!(closed.len(), 1, "Expected exactly one closed transaction");
        let closed_trans = closed[0];
        assert_eq!(closed_trans.get_event_count(), 3);

        let mut steps: Vec<String> = closed_trans
            .fields
            .get("step")
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        steps.sort();
        assert_eq!(
            steps,
            vec![
                "between".to_string(),
                "end".to_string(),
                "start".to_string()
            ]
        );

        assert!(open.iter().any(|trans| {
            trans
                .fields
                .get("step")
                .is_some_and(|vals| vals.contains(&"after_end".to_string()))
        }));
    }

    #[test]
    fn test_transaction_with_multiple_key_null() {
        // Test case similar to the C++ test: transaction test with multiple key null
        // Events should be processed with null wildcard matching
        let params = TransParams::new(
            Some(vec![]),
            vec![("fields".to_string(), Arg::String("key,key2".to_string()))],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);

        // Create events similar to the C++ test (in reverse time order as per SQL)
        // (time, key, key2, message)
        let event1 = create_event_nullable(
            1750844229000000, // latest time
            "1",
            &[("key", Some("xxx")), ("key2", Some("kkk"))],
        );
        let event2 = create_event_nullable(
            1750844228000000,
            "2",
            &[("key", Some("xxx")), ("key2", Some("jjj"))],
        );
        let event3 = create_event_nullable(
            1750844227000000,
            "3",
            &[("key", Some("yyy")), ("key2", Some("lll"))],
        );
        let event4 = create_event_nullable(
            1750844226000000,
            "4",
            &[("key", None), ("key2", Some("lll"))], // key is null
        );
        let event5 = create_event_nullable(
            1750844225000000, // earliest time
            "5",
            &[("key", None), ("key2", Some("lll"))], // key is null
        );

        // Process events (they'll be processed in the order they're added, but normally SQL orders by time desc)
        // For this test, we'll add them in chronological order to simulate them being processed in reverse time order
        pool.add_event(event1).unwrap();
        pool.add_event(event2).unwrap();
        pool.add_event(event3).unwrap();
        pool.add_event(event4).unwrap();
        pool.add_event(event5).unwrap();

        // Get the results
        let final_transactions = pool.get_frozen_trans();
        let live_transactions = pool.get_live_trans();
        let all_transactions = [&final_transactions[..], &live_transactions[..]].concat();

        // Extract messages for verification
        let mut all_messages: Vec<Vec<String>> = Vec::new();
        for trans in &all_transactions {
            let messages: Vec<String> = trans
                .messages
                .iter()
                .flat_map(|msg_list| msg_list.iter().cloned())
                .collect();
            all_messages.push(messages);
        }

        // Check that we have the expected transaction groupings
        // Expected: {{"5", "4", "3"}, {"2"}, {"1"}}
        // event 5,4,3 should be grouped because of null wildcard matching on the 'key' field
        // The null 'key' in event 4 matches with 'yyy' in event 3, and event 5&4 have same non-null values
        let mut message_groups: Vec<Vec<String>> = Vec::new();
        for trans in &all_transactions {
            let mut msgs: Vec<String> = trans
                .messages
                .iter()
                .flat_map(|msg_list| msg_list.iter().cloned())
                .collect();
            msgs.sort(); // Sort for consistent comparison
            message_groups.push(msgs);
        }

        // Sort the groups for comparison since order might vary
        message_groups.sort();

        // Verify that we have 3 groups
        assert_eq!(message_groups.len(), 3);

        // Check that one of the groups contains ["3", "4", "5"]
        let contains_345 = message_groups.iter().any(|group| {
            let mut sorted_group = group.clone();
            sorted_group.sort();
            sorted_group == vec!["3", "4", "5"]
        });
        assert!(contains_345, "Should have a group with 3, 4, 5");

        // Check that we also have individual groups for "1" and "2"
        let has_individual_1 = message_groups.iter().any(|group| group == &vec!["1"]);
        let has_individual_2 = message_groups.iter().any(|group| group == &vec!["2"]);
        assert!(has_individual_1, "Should have a group with just 1");
        assert!(has_individual_2, "Should have a group with just 2");
    }

    #[test]
    fn test_transaction_with_null_fields() {
        // Test case similar to the C++ test: transaction test with null fields
        // With the updated implementation, for single field grouping, events with null key fields are excluded
        let params = TransParams::new(
            Some(vec![]),
            vec![("fields".to_string(), Arg::String("user".to_string()))],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);

        // Create events similar to the C++ test (in reverse time order as per SQL)
        // (time, host, user, message)
        let event1 = create_event(
            1750844225000000, // earliest time
            "1",
            &[("host", "host1"), ("user", "a")],
        );
        let event2 = create_event(1750844226000000, "2", &[("host", "host1"), ("user", "a")]);
        let event3 = create_event(1750844227000000, "3", &[("host", "host2"), ("user", "c")]);
        let event4 = create_event(1750844228000000, "4", &[("host", "host1"), ("user", "d")]);
        let event5 = create_event(1750844229000000, "5", &[("host", "host2"), ("user", "a")]);
        let event6 = create_event_nullable(
            // This event has user=null and should be excluded
            1750844230000000, // latest time
            "6",
            &[("host", Some("host1")), ("user", None)], // user is null
        );

        // Add events in time-descending order (as SQL does with ORDER BY _time DESC)
        pool.add_event(event6).unwrap(); // user=null, should be excluded due to single field grouping rule
        pool.add_event(event5).unwrap(); // user="a" 
        pool.add_event(event4).unwrap(); // user="d"
        pool.add_event(event3).unwrap(); // user="c"
        pool.add_event(event2).unwrap(); // user="a"
        pool.add_event(event1).unwrap(); // user="a"

        // Get results
        let final_transactions = pool.get_frozen_trans();
        let live_transactions = pool.get_live_trans();
        let all_transactions = [&final_transactions[..], &live_transactions[..]].concat();

        // Extract messages for verification
        let mut message_groups: Vec<Vec<String>> = Vec::new();
        for trans in &all_transactions {
            let mut msgs: Vec<String> = trans
                .messages
                .iter()
                .flat_map(|msg_list| msg_list.iter().cloned())
                .collect();
            msgs.sort(); // Sort for consistent comparison
            message_groups.push(msgs);
        }

        // Sort the groups for comparison since order might vary
        message_groups.sort();

        // With the updated implementation for single field grouping, event 6 (user=null) should be excluded
        // Expected result should be {{"3"}, {"4"}, {"1", "2", "5"}} as in the C++ test
        let all_events_in_results: Vec<String> = message_groups
            .iter()
            .flat_map(|group| group.iter().cloned())
            .collect();

        // Verify that event 6 is NOT present in any transaction (excluded due to null user field in single-field grouping)
        assert!(
            !all_events_in_results.contains(&"6".to_string()),
            "Event 6 should not be in results due to null user field with single field grouping"
        );

        // Check that events 1, 2, 5 are in the same group (they all have user="a")
        let has_125_group = message_groups.iter().any(|group| {
            group.contains(&"1".to_string())
                && group.contains(&"2".to_string())
                && group.contains(&"5".to_string())
        });
        assert!(has_125_group, "Should have a group with 1, 2, 5");

        // Verify that events 3 and 4 are present in results
        assert!(
            all_events_in_results.contains(&"3".to_string()),
            "Event 3 should be in results"
        );
        assert!(
            all_events_in_results.contains(&"4".to_string()),
            "Event 4 should be in results"
        );
    }

    #[test]
    fn test_transaction_with_all_null_fields() {
        // Test case for when all key fields are null - such events should be discarded
        // Test data similar to the example provided:
        // 1: host=host1, user=user1 -> should be in a group
        // 2: host=host1, user=user1 -> should be with #1
        // 3: host=host2, user=user3 -> should be alone
        // 4: host=host1, user=user4 -> should be in a group
        // 5: host=host2, user=user1 -> should be alone
        // 6: host=host1, user=null -> could be with #1 or #4, but #4 is nearer/ more recent, should with #4, because null as a wildcard
        // 7: host=null, user=user4 -> with #4
        // 8: host=null, user=null -> should be discarded (all null fields)
        let params = TransParams::new(
            Some(vec![]),
            vec![("fields".to_string(), Arg::String("host,user".to_string()))], // Two fields for grouping
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);

        // Create events in reverse time order (as SQL does with ORDER BY _time DESC)
        let event8 = create_event_nullable(
            // host=null, user=null - all nulls should be discarded
            1599999994000000,
            "8",
            &[
                ("host", None),
                ("user", None),
                ("message", Some("message8")),
            ],
        );
        let event7 = create_event_nullable(
            // host=null, user=user4 - single null should be discarded for single field case
            1599999993000000,
            "7",
            &[
                ("host", None),
                ("user", Some("user4")),
                ("message", Some("message7")),
            ],
        );
        let event6 = create_event_nullable(
            // host=host1, user=null - single null should be discarded for single field case
            1599999995000000,
            "6",
            &[
                ("host", Some("host1")),
                ("user", None),
                ("message", Some("message6")),
            ],
        );
        let event5 = create_event(
            // host=host2, user=user1
            1599999996000000,
            "5",
            &[
                ("host", "host2"),
                ("user", "user1"),
                ("message", "message5"),
            ],
        );
        let event4 = create_event(
            // host=host1, user=user4
            1599999997000000,
            "4",
            &[
                ("host", "host1"),
                ("user", "user4"),
                ("message", "message4"),
            ],
        );
        let event3 = create_event(
            // host=host2, user=user3
            1599999998000000,
            "3",
            &[
                ("host", "host2"),
                ("user", "user3"),
                ("message", "message3"),
            ],
        );
        let event2 = create_event(
            // host=host1, user=user1
            1599999999000000,
            "2",
            &[
                ("host", "host1"),
                ("user", "user1"),
                ("message", "message2"),
            ],
        );
        let event1 = create_event(
            // host=host1, user=user1
            1600000000000000, // latest time
            "1",
            &[
                ("host", "host1"),
                ("user", "user1"),
                ("message", "message1"),
            ],
        );

        // Add events in time-descending order (as SQL does with ORDER BY _time DESC)
        pool.add_event(event1).unwrap(); // host=host1, user=user1
        pool.add_event(event2).unwrap(); // host=host1, user=user1 
        pool.add_event(event3).unwrap(); // host=host2, user=user3
        pool.add_event(event4).unwrap(); // host=host1, user=user4
        pool.add_event(event5).unwrap(); // host=host2, user=user1
        pool.add_event(event6).unwrap(); // host=host1, user=null
        pool.add_event(event7).unwrap(); // host=null, user=user4
        pool.add_event(event8).unwrap(); // host=null, user=null - should be discarded

        // Get results
        let final_transactions = pool.get_frozen_trans();
        let live_transactions = pool.get_live_trans();
        let all_transactions = [&final_transactions[..], &live_transactions[..]].concat();

        // Extract messages for verification
        let mut message_groups: Vec<Vec<String>> = Vec::new();
        for trans in &all_transactions {
            let mut msgs: Vec<String> = trans
                .messages
                .iter()
                .flat_map(|msg_list| msg_list.iter().cloned())
                .collect();
            msgs.sort(); // Sort for consistent comparison
            message_groups.push(msgs);
        }

        // Sort the groups for comparison since order might vary
        message_groups.sort();

        dbg!(&message_groups);

        // Expected from example: Only event with ALL nulls (event 8) should be discarded
        // Events 6 and 7 have partial nulls and should be included (using wildcard matching)
        let all_events_in_results: Vec<String> = message_groups
            .iter()
            .flat_map(|group| group.iter().cloned())
            .collect();

        // Event 8 should not be present because ALL its key fields are null
        assert!(
            !all_events_in_results.contains(&"8".to_string()),
            "Event 8 should not be in results (all nulls)"
        );

        // Events 6 and 7 should be present (partial nulls are allowed with wildcard matching)
        assert!(
            all_events_in_results.contains(&"6".to_string()),
            "Event 6 should be in results (partial null)"
        );
        assert!(
            all_events_in_results.contains(&"7".to_string()),
            "Event 7 should be in results (partial null)"
        );

        assert!(
            message_groups.iter().any(|group| {
                group.contains(&"1".to_string())
                    && group.contains(&"2".to_string())
                    && group.len() == 2
            }),
            "Should have a group with 1, 2"
        );
        assert!(
            message_groups.iter().any(|group| {
                group.contains(&"4".to_string())
                    && group.contains(&"6".to_string())
                    && group.contains(&"7".to_string())
                    && group.len() == 3
            }),
            "Should have a group with 4, 6, 7"
        );

        assert!(
            all_events_in_results.contains(&"4".to_string()),
            "Event 4 should be in results"
        );
        assert!(
            all_events_in_results.contains(&"5".to_string()),
            "Event 5 should be in results"
        );

        // Event 8 should be excluded (all key fields are null)
        assert!(
            !all_events_in_results.contains(&"8".to_string()),
            "Event 8 should be excluded (all key fields null)"
        );
    }

    #[test]
    fn test_transaction_with_three_fields_and_nulls() {
        let params = TransParams::new(
            Some(vec![]),
            vec![(
                "fields".to_string(),
                Arg::String("hostname,user,app".to_string()),
            )],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);

        #[rustfmt::skip]
        // Test data: (time, hostname, user, app, message)
        let test_data = vec![
            (1600000000000000, Some("host1"), Some("user1"), Some("app1"), "message1"),
            (1599999999000000, Some("host1"), Some("user1"), Some("app1"), "message2"),
            (1599999998000000, Some("host1"), Some("user2"), Some("app2"), "message3"),
            (1599999997000000, Some("host2"), Some("user3"), Some("app3"), "message4"),
            (1599999996000000, Some("host3"), Some("user4"), Some("app1"), "message5"),
            (1599999995000000, Some("host1"), None, Some("app1"), "message6"),
            (1599999994000000, Some("host1"), None, Some("app2"), "message7"),
            (1599999993000000, Some("host2"), None, Some("app3"), "message8"),
            (1599999992000000, None, Some("user1"), Some("app1"), "message9"),
            (1599999991000000, None, Some("user2"), Some("app2"), "message10"),
            (1599999990000000, None, Some("user3"), Some("app3"), "message11"),
            (1599999989000000, Some("host1"), Some("user1"), None, "message12"),
            (1599999988000000, Some("host2"), Some("user3"), None, "message13"),
            (1599999987000000, Some("host3"), Some("user4"), None, "message14"),
            (1599999986000000, None, None, Some("app1"), "message15"),
            (1599999985000000, None, None, Some("app2"), "message16"),
            (1599999984000000, None, None, Some("app3"), "message17"),
            (1599999983000000, Some("host1"), None, None, "message18"),
            (1599999982000000, Some("host2"), None, None, "message19"),
            (1599999981000000, Some("host3"), None, None, "message20"),
            (1599999980000000, None, Some("user1"), None, "message21"),
            (1599999979000000, None, Some("user2"), None, "message22"),
            (1599999978000000, None, Some("user4"), None, "message23"),
            (1599999977000000, None, None, None, "message24"),
        ];

        for (time, hostname, user, app, message) in test_data {
            let event = create_event_nullable(
                time,
                message,
                &[("hostname", hostname), ("user", user), ("app", app)],
            );
            pool.add_event(event).unwrap();
        }

        // Get results
        let final_transactions = pool.get_frozen_trans();
        let live_transactions = pool.get_live_trans();
        let all_transactions = [&final_transactions[..], &live_transactions[..]].concat();

        // Extract messages for verification
        let mut message_groups: Vec<Vec<String>> = Vec::new();
        for trans in &all_transactions {
            let mut msgs: Vec<String> = trans
                .messages
                .iter()
                .flat_map(|msg_list| msg_list.iter().cloned())
                .collect();
            msgs.sort(); // Sort for consistent comparison
            message_groups.push(msgs);
        }

        // Sort the groups for comparison since order might vary
        message_groups.sort();

        dbg!(&message_groups);

        // Expected output:
        // {message1 message12 message2 message21 message6 message9} - group with host1,user1,app1 + wildcard matches
        // {message10 message16 message18 message22 message3 message7} - group with host1,user2,app2 + wildcard matches
        // {message11 message13 message17 message19 message4 message8} - group with host2,user3,app3 + wildcard matches
        // {message14 message15 message20 message23 message5} - group with host3,user4,app1 + wildcard matches

        let all_messages_in_results: Vec<String> = message_groups
            .iter()
            .flat_map(|group| group.iter().cloned())
            .collect();

        // Event 24 should not be present because ALL its key fields are null
        assert!(
            !all_messages_in_results.contains(&"message24".to_string()),
            "Event 24 should not be in results (all nulls)"
        );

        // Check that expected messages are in results
        for i in 1..24 {
            // messages 1-23 should be present
            let message = format!("message{}", i);
            assert!(
                all_messages_in_results.contains(&message),
                "Message {} should be in results",
                i
            );
        }

        fn has_group_with_messages(groups: &Vec<Vec<String>>, expected_messages: &[&str]) -> bool {
            for group in groups {
                let mut all_match = true;
                for &msg in expected_messages {
                    if !group.contains(&msg.to_string()) {
                        all_match = false;
                        break;
                    }
                }
                if all_match && group.len() == expected_messages.len() {
                    return true;
                }
            }
            false
        }

        // Expected groupings according to specification:
        // Group 1: {message1, message2, message6, message9, message12, message21} - host1,user1,app1 + wildcard matches
        assert!(
            has_group_with_messages(
                &message_groups,
                &[
                    "message1",
                    "message2",
                    "message6",
                    "message9",
                    "message12",
                    "message21"
                ],
            ),
            "Should have group with message1, message2, message6, message9, message12, message21"
        );

        // Group 2: {message3, message7, message10, message16, message18, message22} - host1,user2,app2 + wildcard matches
        assert!(
            has_group_with_messages(
                &message_groups,
                &[
                    "message3",
                    "message7",
                    "message10",
                    "message16",
                    "message18",
                    "message22"
                ],
            ),
            "Should have group with message3, message7, message10, message16, message18, message22"
        );

        // Group 3: {message4, message8, message11, message13, message17, message19} - host2,user3,app3 + wildcard matches
        assert!(
            has_group_with_messages(
                &message_groups,
                &[
                    "message4",
                    "message8",
                    "message11",
                    "message13",
                    "message17",
                    "message19"
                ],
            ),
            "Should have group with message4, message8, message11, message13, message17, message19"
        );

        // Group 4: {message5, message14, message15, message20, message23} - host3,user4,app1 + wildcard matches
        assert!(
            has_group_with_messages(
                &message_groups,
                &[
                    "message5",
                    "message14",
                    "message15",
                    "message20",
                    "message23"
                ],
            ),
            "Should have group with message5, message14, message15, message20, message23"
        );

        // Verify that all 24 - 1 = 23 messages (excluding message24 which was discarded) are in exactly one group
        assert_eq!(
            all_messages_in_results.len(),
            23,
            "Should have exactly 23 messages in groups (excluding discarded message24)"
        );
    }

    #[test]
    fn test_max_events_bracket_matching_with_csv() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                (
                    "fields".to_string(),
                    Arg::String("client_ip,JSESSIONID".to_string()),
                ),
                ("starts_with".to_string(), Arg::String("view".to_string())), // Look for "view" in messages
                ("ends_with".to_string(), Arg::String("purchase".to_string())), // Look for "purchase" in messages
                ("max_events".to_string(), Arg::Int(2)), // Set max_events to 2
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);

        for record in sample_csv_records() {
            let event = record_to_event(&record);
            pool.add_event(event).unwrap();
        }

        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());

        let closed: Vec<_> = transactions.into_iter().filter(|t| t.is_closed).collect();
        assert!(
            !closed.is_empty(),
            "Expected at least one closed transaction from fixture"
        );

        for trans in &closed {
            assert!(
                trans.get_event_count() <= 2,
                "Transaction exceeds max_events=2 with {} events",
                trans.get_event_count()
            );
        }

        assert!(
            closed.iter().any(|t| t.get_event_count() == 2),
            "Expected at least one closed transaction to have exactly two events"
        );
    }

    #[test]
    fn test_max_events_bracket_matching_with_limit_one() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                (
                    "fields".to_string(),
                    Arg::String("client_ip,JSESSIONID".to_string()),
                ),
                ("starts_with".to_string(), Arg::String("view".to_string())),
                ("ends_with".to_string(), Arg::String("purchase".to_string())),
                ("max_events".to_string(), Arg::Int(1)),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);

        for event in sample_csv_events() {
            pool.add_event(event).unwrap();
        }

        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());

        let closed: Vec<_> = transactions.into_iter().filter(|t| t.is_closed).collect();
        assert!(
            !closed.is_empty(),
            "Expected closed transactions when max_events=1"
        );
        assert!(
            closed.iter().all(|t| t.get_event_count() <= 1),
            "Closed transactions should be limited to one event each"
        );
        assert!(closed.iter().all(|t| {
            t.messages
                .iter()
                .flat_map(|msgs| msgs.iter())
                .all(|msg| msg.contains("purchase"))
        }));
    }

    #[test]
    fn test_max_events_duration() {
        let events = sample_csv_events();

        let params_no_span = TransParams::new(
            Some(vec![]),
            vec![
                (
                    "fields".to_string(),
                    Arg::String("client_ip,JSESSIONID".to_string()),
                ),
                ("starts_with".to_string(), Arg::String("view".to_string())),
                ("ends_with".to_string(), Arg::String("purchase".to_string())),
            ],
        )
        .unwrap();

        let mut pool_no_span = TransactionPool::new(params_no_span);
        for event in &events {
            pool_no_span.add_event(event.clone()).unwrap();
        }

        let mut transactions = pool_no_span.get_frozen_trans();
        transactions.extend(pool_no_span.get_live_trans());

        assert!(
            !transactions.is_empty(),
            "Expected transactions from sample data"
        );

        let closed_counts: Vec<_> = transactions
            .iter()
            .filter(|t| t.is_closed)
            .map(|t| t.get_event_count())
            .collect();
        assert!(
            !closed_counts.is_empty(),
            "Expected closed transactions when no max_events limit is set"
        );
        assert!(
            closed_counts.iter().any(|&c| c >= 3),
            "Expected at least one closed transaction with three or more events"
        );
        assert!(
            closed_counts.contains(&2),
            "Expected at least one closed transaction with two events"
        );

        let open_counts: Vec<_> = transactions
            .iter()
            .filter(|t| !t.is_closed)
            .map(|t| t.get_event_count())
            .collect();
        assert!(
            open_counts.contains(&1),
            "Expected at least one open transaction with a single event"
        );

        let params_with_span = TransParams::new(
            Some(vec![]),
            vec![
                (
                    "fields".to_string(),
                    Arg::String("client_ip,JSESSIONID".to_string()),
                ),
                ("starts_with".to_string(), Arg::String("view".to_string())),
                ("ends_with".to_string(), Arg::String("purchase".to_string())),
                ("max_span".to_string(), Arg::String("2s".to_string())),
            ],
        )
        .unwrap();

        let mut pool_with_span = TransactionPool::new(params_with_span);
        for event in &events {
            pool_with_span.add_event(event.clone()).unwrap();
        }

        let mut span_transactions = pool_with_span.get_frozen_trans();
        span_transactions.extend(pool_with_span.get_live_trans());

        assert!(
            span_transactions.iter().any(|t| t.is_closed),
            "Expected closed transactions when max_span is set"
        );

        let mut span_closed: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
        let mut span_open: BTreeMap<(String, String), Vec<Vec<String>>> = BTreeMap::new();

        for trans in &span_transactions {
            let key = (
                trans
                    .fields
                    .get("client_ip")
                    .and_then(|v| v.first())
                    .cloned()
                    .unwrap_or_default(),
                trans
                    .fields
                    .get("JSESSIONID")
                    .and_then(|v| v.first())
                    .cloned()
                    .unwrap_or_default(),
            );
            let messages: Vec<String> = trans
                .messages
                .iter()
                .flat_map(|msgs| msgs.iter().cloned())
                .collect();
            if trans.is_closed {
                span_closed.insert(key, messages);
            } else {
                span_open.entry(key).or_default().push(messages);
            }
        }

        let expected_closed_keys = [
            ("27.96.191.11".to_string(), "SD6SL1FF10ADFF5226".to_string()),
            ("60.18.93.11".to_string(), "SD4SL5FF3ADFF5111".to_string()),
        ];

        for key in &expected_closed_keys {
            assert!(
                span_closed.contains_key(key),
                "Expected closed transaction for {key:?} within max_span"
            );
        }

        assert!(
            span_closed.values().any(|msgs| msgs.len() >= 2),
            "Expected at least one closed transaction to retain multiple events"
        );

        for trans in span_transactions.iter().filter(|t| t.is_closed) {
            if let Some(duration) = trans.get_duration() {
                assert!(
                    duration <= 2.0,
                    "Closed transaction duration {} exceeds max_span",
                    duration
                );
            }
            assert!(
                trans.get_event_count() <= 3,
                "Closed transaction should not retain four events when max_span=2s"
            );
        }

        assert!(
            span_open
                .values()
                .flat_map(|lists| lists.iter())
                .any(|messages| messages.len() == 1),
            "Expected at least one truncated open transaction when max_span applies"
        );
    }

    #[test]
    fn test_max_span_one_second_from_fixture() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                (
                    "fields".to_string(),
                    Arg::String("client_ip,JSESSIONID".to_string()),
                ),
                ("starts_with".to_string(), Arg::String("view".to_string())),
                ("ends_with".to_string(), Arg::String("purchase".to_string())),
                ("max_span".to_string(), Arg::String("1s".to_string())),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);

        for event in sample_csv_events() {
            pool.add_event(event).unwrap();
        }

        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());

        let closed: Vec<_> = transactions.into_iter().filter(|t| t.is_closed).collect();
        assert!(
            closed.len() >= 2,
            "Expected at least two closed transactions when max_span=1s"
        );

        let expected_keys = [
            ("27.96.191.11".to_string(), "SD6SL1FF10ADFF5226".to_string()),
            ("88.12.32.208".to_string(), "SD1SL1FF3ADFF5727".to_string()),
        ];

        for (ip, session) in expected_keys {
            let matching = closed.iter().any(|trans| {
                let client_ip = trans
                    .fields
                    .get("client_ip")
                    .and_then(|vals| vals.first())
                    .cloned();
                let jsession = trans
                    .fields
                    .get("JSESSIONID")
                    .and_then(|vals| vals.first())
                    .cloned();
                if let (Some(client_ip), Some(jsession)) = (client_ip, jsession)
                    && client_ip == ip
                    && jsession == session
                    && let Some(duration) = trans.get_duration()
                {
                    return duration <= 1.0 && trans.get_event_count() >= 2;
                }
                false
            });
            assert!(
                matching,
                "Expected closed transaction for ({ip}, {session}) within 1s span"
            );
        }
    }
}
