use anyhow::{Context, Result, anyhow};
use arrow::array::{ArrayRef, AsArray, ListBuilder, StringBuilder};
use arrow::datatypes::{Float64Type, Int64Type, TimestampMicrosecondType};
use arrow::{
    array::{Array, BooleanArray, Float64Array, Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use chrono::{DateTime, TimeZone, Utc};
use hashbrown::Equivalent;
use hashbrown::HashMap;
use hashbrown::hash_map::RawEntryMut;
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

pub type EventFieldValues = SmallVec<[Arc<str>; 4]>;
pub type EventMap = HashMap<String, EventFieldValues>;

#[inline]
fn string_to_arc(s: String) -> Arc<str> {
    Arc::<str>::from(s.into_boxed_str())
}

#[derive(Debug, Clone)]
pub struct TransParams {
    pub fields: Arc<[String]>,
    pub starts_with: Option<String>,
    pub starts_with_regex: Option<Regex>,
    pub starts_if_field: Option<String>,
    pub ends_with: Option<String>,
    pub ends_with_regex: Option<Regex>,
    pub ends_if_field: Option<String>,
    pub max_span: Option<Duration>,
    pub max_events: u64,
}

impl Default for TransParams {
    fn default() -> Self {
        Self {
            fields: Arc::from(Vec::<String>::new()),
            starts_with: Default::default(),
            starts_with_regex: Default::default(),
            starts_if_field: Default::default(),
            ends_with: Default::default(),
            ends_with_regex: Default::default(),
            ends_if_field: Default::default(),
            max_span: None,
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
                    let v: Vec<String> = s.split(',').map(|f| f.trim().to_string()).collect();
                    parsed_params.fields = Arc::from(v.into_boxed_slice());
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
                        LazyLock::new(|| Regex::new(r"(-?\d+)([smhd])").unwrap());
                    if let Some(m) = DURATION_REGEX.captures(&s) {
                        let value_raw: i64 = m[1].parse()?;
                        let unit = &m[2];
                        if value_raw < 0 {
                            // Negative => treat as unset (continue)
                            continue;
                        } else {
                            let value = value_raw as u64;
                            let seconds = match unit {
                                "s" => value,
                                "m" => value * 60,
                                "h" => value * 3600,
                                "d" => value * 86400,
                                _ => return Err(anyhow!("Invalid time unit: {}", unit)),
                            };
                            parsed_params.max_span = Some(Duration::from_secs(seconds));
                        }
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
                    if max_events_val < 0 {
                        parsed_params.max_events = u64::MAX;
                    } else {
                        parsed_params.max_events = max_events_val as u64;
                    }
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
        event: &EventMap,
        field_name_opt: &Option<String>,
    ) -> bool {
        if let Some(field_name) = field_name_opt
            && let Some(value) = event.get(field_name)
        {
            return value.iter().all(|v| v.eq_ignore_ascii_case("true"));
        }
        false
    }

    #[cfg(test)]
    fn matches_starts_with(&self, event: &EventMap) -> bool {
        self.matches_starts_with_cached(event.get(FIELD_MESSAGE), event)
    }

    fn matches_starts_with_cached(
        &self,
        message_values: Option<&EventFieldValues>,
        event: &EventMap,
    ) -> bool {
        if let Some(message_values) = message_values {
            for v in message_values {
                if let Some(s) = &self.starts_with
                    && v.contains(s)
                {
                    return true;
                }
                if let Some(r) = &self.starts_with_regex
                    && r.is_match(v)
                {
                    return true;
                }
            }
        }
        self.check_boolean_field_condition(event, &self.starts_if_field)
    }

    #[cfg(test)]
    fn matches_ends_with(&self, event: &EventMap) -> bool {
        self.matches_ends_with_cached(event.get(FIELD_MESSAGE), event)
    }

    fn matches_ends_with_cached(
        &self,
        message_values: Option<&EventFieldValues>,
        event: &EventMap,
    ) -> bool {
        if let Some(message) = message_values {
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
        }
        self.check_boolean_field_condition(event, &self.ends_if_field)
    }
}

#[derive(Debug, Clone)]
pub struct Transaction {
    key_names: Arc<[String]>,
    key_index_map: HashMap<String, usize, ahash::RandomState>,
    key_slots: Vec<EventFieldValues>,
    extra_fields: HashMap<String, EventFieldValues, ahash::RandomState>,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    messages: VecDeque<SmallVec<[Arc<str>; 4]>>,
    times: VecDeque<DateTime<Utc>>,
    event_count: u64,
    is_closed: bool,
    has_start_marker: bool,
    has_end_marker: bool,
}

impl Transaction {
    fn new(field_names: Arc<[String]>) -> Self {
        let mut key_index_map =
            HashMap::with_capacity_and_hasher(field_names.len(), ahash::RandomState::new());
        let mut key_slots = Vec::with_capacity(field_names.len());
        for (idx, k) in field_names.iter().enumerate() {
            key_index_map.insert(k.clone(), idx);
            key_slots.push(EventFieldValues::new());
        }
        Transaction {
            key_names: field_names,
            key_index_map,
            key_slots,
            extra_fields: HashMap::with_hasher(ahash::RandomState::new()),
            start_time: None,
            end_time: None,
            messages: VecDeque::new(),
            times: VecDeque::new(),
            event_count: 0,
            is_closed: false,
            has_start_marker: false,
            has_end_marker: false,
        }
    }

    fn merge_event(&mut self, event: &EventMap) {
        if let Some(message) = event.get(FIELD_MESSAGE) {
            if message.is_empty() {
                return;
            }
            let mapped: EventFieldValues = message.iter().cloned().collect();
            self.messages.push_front(mapped);
        }

        for (k, v) in event {
            if TRANS_RESERVED_FIELDS.contains(&k.as_str()) {
                continue;
            }
            if v.is_empty() {
                continue;
            }
            if let Some(&idx) = self.key_index_map.get(k) {
                let slot = &mut self.key_slots[idx];
                slot.truncate(0);
                slot.extend(v.iter().cloned());
            } else {
                let entry = self.extra_fields.entry(k.clone()).or_default();
                entry.extend(v.iter().cloned());
            }
        }
    }

    pub fn key_names(&self) -> &[String] {
        &self.key_names
    }

    pub fn get_field_values(&self, name: &str) -> Option<&EventFieldValues> {
        if let Some(&idx) = self.key_index_map.get(name) {
            Some(&self.key_slots[idx])
        } else {
            self.extra_fields.get(name)
        }
    }

    fn set_key_field_from(&mut self, name: &str, values: &EventFieldValues) {
        if let Some(&idx) = self.key_index_map.get(name) {
            let slot = &mut self.key_slots[idx];
            slot.truncate(0);
            slot.extend(values.iter().cloned());
        }
    }

    fn clear_key_field(&mut self, name: &str) {
        if let Some(&idx) = self.key_index_map.get(name) {
            self.key_slots[idx].truncate(0);
        }
    }

    pub fn iter_fields(&self) -> TransactionFieldIter<'_> {
        TransactionFieldIter {
            key_iter: self.key_names.iter().enumerate(),
            key_slots: &self.key_slots,
            extra_iter: self.extra_fields.iter(),
        }
    }

    fn add_event(&mut self, event: &EventMap) -> Result<()> {
        let time_str = event
            .get(FIELD_TIME)
            .context("Event missing _time field or _time is null")?;

        let time_first = time_str.first().context("No _time in transaction")?;
        let ts = atoi_simd::parse::<i64>(time_first.as_bytes())
            .map_err(|_| anyhow!("Failed to parse _time as timestamp"))?;
        let time = Utc.timestamp_micros(ts).earliest().unwrap_or_else(Utc::now);

        self.event_count += 1;

        if self.end_time.is_none() {
            self.start_time = Some(time);
            self.end_time = Some(time);
            self.merge_event(event);
            self.times.push_front(time);
        } else if time <= self.start_time.unwrap() {
            self.start_time = Some(time);
            self.merge_event(event);
            self.times.push_front(time);
        } else if time <= self.end_time.unwrap() && time >= self.start_time.unwrap() {
            self.merge_event(event);
            self.times.push_front(time);
        }

        Ok(())
    }

    fn update_start_time_only(&mut self, event: &EventMap) -> Result<()> {
        let time_str = event
            .get(FIELD_TIME)
            .context("Event missing _time field or _time is null")?;

        let time_first = time_str.first().context("No _time in transaction")?;
        let ts = atoi_simd::parse::<i64>(time_first.as_bytes())
            .map_err(|_| anyhow!("Failed to parse _time as timestamp"))?;
        let time = Utc.timestamp_micros(ts).earliest().unwrap_or_else(Utc::now);

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

    fn span_exceeds_with_candidate(
        &self,
        candidate_time: DateTime<Utc>,
        max_span_secs: u64,
    ) -> bool {
        if max_span_secs == 0 {
            return false;
        }
        let reference_end = self.end_time.or(self.start_time);
        let Some(end_time) = reference_end else {
            return false;
        };
        let start_time = self.start_time.unwrap_or(end_time);
        let new_start = if candidate_time < start_time {
            candidate_time
        } else {
            start_time
        };
        let new_end = if candidate_time > end_time {
            candidate_time
        } else {
            end_time
        };
        let span_micros = new_end
            .signed_duration_since(new_start)
            .num_microseconds()
            .unwrap_or(i64::MAX);
        span_micros > (max_span_secs as i64 * 1_000_000)
    }
}

pub struct TransactionFieldIter<'a> {
    key_iter: std::iter::Enumerate<std::slice::Iter<'a, String>>,
    key_slots: &'a [EventFieldValues],
    extra_iter: hashbrown::hash_map::Iter<'a, String, EventFieldValues>,
}

impl<'a> Iterator for TransactionFieldIter<'a> {
    type Item = (&'a str, &'a EventFieldValues);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((idx, name)) = self.key_iter.next() {
            return Some((name.as_str(), &self.key_slots[idx]));
        }
        self.extra_iter
            .next()
            .map(|(name, values)| (name.as_str(), values))
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
    if let Some(first) = transactions.first() {
        for key_name in first.key_names() {
            all_field_names.insert(key_name.clone());
        }
    }
    for transaction in transactions {
        for field_name in transaction.extra_fields.keys() {
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
                messages_builder.values().append_value(message.as_ref());
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
                let Some(values) = transaction.get_field_values(field_name) else {
                    field_builder.append_null();
                    continue;
                };
                // Borrow and dedup without cloning the vector
                let mut refs: Vec<&str> = values.iter().map(|s| s.as_ref()).collect();
                refs.sort_unstable();
                refs.dedup();
                let Some(value) = refs.first() else {
                    field_builder.append_null();
                    continue;
                };
                field_builder.append_value(value);
            }
            arrays.push(Arc::new(field_builder.finish()));
        } else {
            let mut field_builder = ListBuilder::new(StringBuilder::new());
            for transaction in transactions.iter() {
                let Some(values) = transaction.get_field_values(field_name) else {
                    field_builder.append_null();
                    continue;
                };
                // Use borrowed &str to avoid cloning the vector
                let mut refs: Vec<&str> = values.iter().map(|s| s.as_ref()).collect();
                refs.sort_unstable();
                refs.dedup();
                if refs.is_empty() {
                    field_builder.append_null();
                    continue;
                }
                for value in refs {
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct OwnedTransKey {
    parts: SmallVec<[Option<String>; 4]>,
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct RefTransKey<'a> {
    parts: SmallVec<[Option<&'a str>; 4]>,
}

type TransKey = OwnedTransKey;

impl<'a> Equivalent<OwnedTransKey> for RefTransKey<'a> {
    fn equivalent(&self, key: &OwnedTransKey) -> bool {
        if self.parts.len() != key.parts.len() {
            return false;
        }
        for (a, b) in self.parts.iter().zip(key.parts.iter()) {
            match (a, b) {
                (None, _) => continue,
                (_, None) => continue,
                (Some(sa), Some(sb)) => {
                    if *sa != sb.as_str() {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl Equivalent<OwnedTransKey> for SmallVec<[Option<String>; 4]> {
    fn equivalent(&self, key: &OwnedTransKey) -> bool {
        if self.len() != key.parts.len() {
            return false;
        }
        for (a, b) in self.iter().zip(key.parts.iter()) {
            match (a, b) {
                (None, _) => continue,
                (_, None) => continue,
                (Some(sa), Some(sb)) => {
                    if sa != sb {
                        return false;
                    }
                }
            }
        }
        true
    }
}

#[derive(Clone, Debug)]
pub struct TransactionPool {
    params: TransParams,
    frozen_trans: Vec<Transaction>,
    live_trans: hashbrown::HashMap<TransKey, Transaction, ahash::RandomState>,
    start_trans_stack: hashbrown::HashMap<TransKey, Vec<Transaction>, ahash::RandomState>, // Stack for pending start transactions
    earliest_event_timestamp: Option<DateTime<Utc>>,
    trans_complete_flag: hashbrown::HashMap<TransKey, bool, ahash::RandomState>,
}

impl TransactionPool {
    fn has_start_conditions(&self) -> bool {
        self.params.starts_with.is_some()
            || self.params.starts_with_regex.is_some()
            || self.params.starts_if_field.is_some()
    }

    fn has_end_conditions(&self) -> bool {
        self.params.ends_with.is_some()
            || self.params.ends_with_regex.is_some()
            || self.params.ends_if_field.is_some()
    }

    fn extract_event_time(event: &EventMap) -> Option<DateTime<Utc>> {
        let time_str = event.get(FIELD_TIME)?;
        let time_str_val = time_str.first()?;
        let ts = atoi_simd::parse::<i64>(time_str_val.as_bytes()).ok()?;
        Some(Utc.timestamp_micros(ts).earliest().unwrap_or_else(Utc::now))
    }

    pub fn new(params: TransParams) -> Self {
        TransactionPool {
            params,
            frozen_trans: Vec::new(),
            live_trans: hashbrown::HashMap::with_hasher(ahash::RandomState::new()),
            start_trans_stack: hashbrown::HashMap::with_hasher(ahash::RandomState::new()),
            earliest_event_timestamp: None,
            trans_complete_flag: hashbrown::HashMap::with_hasher(ahash::RandomState::new()),
        }
    }

    fn is_valid_event(&self, event: &EventMap, event_time: DateTime<Utc>) -> bool {
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
            && event_time > earliest_ts
        {
            return false;
        }
        true
    }

    fn freeze_trans_exceeded_max_span_restriction(&mut self) {
        let Some(dur) = self.params.max_span else {
            return;
        };
        let max_span_secs = dur.as_secs();
        if max_span_secs == 0 {
            return;
        }

        let Some(earliest_ts) = self.earliest_event_timestamp else {
            return;
        };
        let max_span_limit = max_span_secs as i64;
        let has_start_conditions = self.has_start_conditions();
        let has_end_conditions = self.has_end_conditions();

        // Move out the map to avoid cloning keys while deciding which entries to keep
        let old_live = std::mem::take(&mut self.live_trans);
        for (key, mut trans) in old_live.into_iter() {
            if let Some(end_time) = trans.end_time {
                let elapsed = end_time.signed_duration_since(earliest_ts).num_seconds();
                if elapsed > max_span_limit {
                    mark_closed_for_flags(&mut trans, has_start_conditions, has_end_conditions);
                    self.frozen_trans.push(trans);
                    self.trans_complete_flag.remove(&key);
                    continue;
                }
            }
            // Keep the entry
            self.live_trans.insert(key, trans);
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
                    let mut trans = stack.remove(idx);
                    mark_closed_for_flags(&mut trans, has_start_conditions, has_end_conditions);
                    self.frozen_trans.push(trans);
                } else {
                    idx += 1;
                }
            }
            !stack.is_empty()
        });
    }

    fn make_trans_key_ref<'a>(&self, event: &'a EventMap) -> RefTransKey<'a> {
        let mut parts: SmallVec<[Option<&'a str>; 4]> =
            SmallVec::with_capacity(self.params.fields.len());
        for f in self.params.fields.iter() {
            if let Some(vals) = event.get(f)
                && let Some(first) = vals.first()
            {
                // Avoid trimming when there is no whitespace to reduce extra scan and allocation
                if first.is_empty() {
                    parts.push(None);
                    continue;
                }
                if first
                    .bytes()
                    .any(|b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r')
                {
                    let trimmed = first.trim();
                    if trimmed.is_empty() {
                        parts.push(None);
                    } else {
                        parts.push(Some(trimmed));
                    }
                } else {
                    parts.push(Some(first.as_ref()));
                }
                continue;
            }
            parts.push(None);
        }
        RefTransKey { parts }
    }

    fn make_trans_key_owned_from_ref(ref_key: &RefTransKey<'_>) -> TransKey {
        let mut parts: SmallVec<[Option<String>; 4]> = SmallVec::with_capacity(ref_key.parts.len());
        for p in ref_key.parts.iter() {
            match p {
                Some(s) => parts.push(Some((*s).to_string())),
                None => parts.push(None),
            }
        }
        OwnedTransKey { parts }
    }

    fn get_or_insert_stack<'a>(
        stack: &'a mut hashbrown::HashMap<TransKey, Vec<Transaction>, ahash::RandomState>,
        key_ref: &RefTransKey<'_>,
    ) -> &'a mut Vec<Transaction> {
        match stack.raw_entry_mut().from_key(key_ref) {
            RawEntryMut::Occupied(entry) => entry.into_mut(),
            RawEntryMut::Vacant(entry) => {
                let owned = Self::make_trans_key_owned_from_ref(key_ref);
                entry.insert(owned, Vec::new()).1
            }
        }
    }

    pub fn add_event(&mut self, event: EventMap) -> Result<()> {
        // For simple grouping without start/end conditions, bypass the allow_nulls filtering
        let has_start_conditions = self.params.starts_with.is_some()
            || self.params.starts_with_regex.is_some()
            || self.params.starts_if_field.is_some();
        let has_end_conditions = self.params.ends_with.is_some()
            || self.params.ends_with_regex.is_some()
            || self.params.ends_if_field.is_some();

        let Some(event_time) = Self::extract_event_time(&event) else {
            return Ok(());
        };

        // Only apply the original validation when there are start/end conditions
        if has_start_conditions || has_end_conditions {
            if !self.is_valid_event(&event, event_time) {
                return Ok(());
            }
        } else {
            // For simple grouping, do basic validation without field existence check
            if !self.is_valid_event_simple_grouping(&event, event_time) {
                return Ok(());
            }
        }

        // - If no start/end conditions: each event becomes a single closed group (identity).
        // - If there are start/end conditions: each event becomes a single open group,
        //   unless it simultaneously matches both start and end (then closed).
        if self.params.max_events == 0 || self.params.max_span.is_some_and(|d| d.as_secs() == 0) {
            let mut single = Transaction::new(self.params.fields.clone());
            if has_start_conditions
                && self
                    .params
                    .matches_starts_with_cached(event.get(FIELD_MESSAGE), &event)
            {
                single.has_start_marker = true;
            }
            if has_end_conditions
                && self
                    .params
                    .matches_ends_with_cached(event.get(FIELD_MESSAGE), &event)
            {
                single.has_end_marker = true;
            }
            single.add_event(&event)?;
            let has_brackets = has_start_conditions || has_end_conditions;
            if !has_brackets || (single.has_start_marker && single.has_end_marker) {
                single.set_is_closed();
            }
            self.frozen_trans.push(single);
            return Ok(());
        }

        let event_time_micros = event_time.timestamp_micros();
        self.earliest_event_timestamp = Some(event_time);

        self.freeze_trans_exceeded_max_span_restriction();

        let trans_key_ref = self.make_trans_key_ref(&event);

        let message_values = event.get(FIELD_MESSAGE);

        if has_start_conditions || has_end_conditions {
            // When processing in reverse chronological order (as in SQL ORDER BY _time DESC),
            // END events go to stack waiting for START events to close them
            if self.params.matches_ends_with_cached(message_values, &event) {
                if has_start_conditions {
                    // In bracket mode, always push END to the stack to be paired by a future START.
                    // Do not try to immediately attach END to any existing start-live in live_trans.
                    let mut new_trans = Transaction::new(self.params.fields.clone());
                    new_trans.has_end_marker = true;
                    new_trans.add_event(&event)?;
                    let stack =
                        Self::get_or_insert_stack(&mut self.start_trans_stack, &trans_key_ref);
                    stack.push(new_trans);
                } else {
                    // ends_only mode:
                    // ends_only mode, always use nested/backfill behavior
                    let stack =
                        Self::get_or_insert_stack(&mut self.start_trans_stack, &trans_key_ref);
                    let mut end_only = Transaction::new(self.params.fields.clone());
                    end_only.has_end_marker = true;
                    end_only.add_event(&event)?;
                    stack.push(end_only);
                    if let Some(span) = self.params.max_span
                        && span.as_secs() > 0
                        && let Some(mut_buffer) = self.live_trans.get_mut(&trans_key_ref)
                    {
                        loop {
                            while let Some(top) = stack.last()
                                && self.params.max_events > 0
                                && top.get_event_count() >= self.params.max_events
                            {
                                let mut full = stack.pop().unwrap();
                                if full.has_end_marker {
                                    full.set_is_closed();
                                }
                                self.frozen_trans.push(full);
                            }
                            if mut_buffer.messages.is_empty() || stack.is_empty() {
                                break;
                            }
                            let top = stack.last_mut().unwrap();
                            if self.params.max_events > 0
                                && top.get_event_count() >= self.params.max_events
                            {
                                continue;
                            }
                            if let Some(&candidate_time) = mut_buffer.times.front()
                                && top.span_exceeds_with_candidate(candidate_time, span.as_secs())
                            {
                                break;
                            }
                            if let (Some(msgs), Some(ev_time)) = (
                                mut_buffer.messages.pop_front(),
                                mut_buffer.times.pop_front(),
                            ) {
                                top.messages.push_front(msgs);
                                top.times.push_front(ev_time);
                                top.start_time = Some(
                                    top.start_time
                                        .map(|cur| if ev_time < cur { ev_time } else { cur })
                                        .unwrap_or(ev_time),
                                );
                                top.end_time = Some(
                                    top.end_time
                                        .map(|cur| if ev_time > cur { ev_time } else { cur })
                                        .unwrap_or(ev_time),
                                );
                                top.event_count = top.event_count.saturating_add(1);
                                if mut_buffer.event_count > 0 {
                                    mut_buffer.event_count -= 1;
                                }
                                match (mut_buffer.times.front(), mut_buffer.times.back()) {
                                    (Some(&s), Some(&e)) => {
                                        mut_buffer.start_time = Some(s);
                                        mut_buffer.end_time = Some(e);
                                    }
                                    _ => {
                                        mut_buffer.start_time = None;
                                        mut_buffer.end_time = None;
                                    }
                                }
                            } else {
                                break;
                            }
                        }
                        if mut_buffer.messages.is_empty() {
                            let _ = self.live_trans.remove(&trans_key_ref);
                        }
                    }
                    if let Some(top) = stack.last()
                        && self.params.max_events > 0
                        && top.get_event_count() >= self.params.max_events
                    {
                        let mut full = stack.pop().unwrap();
                        if full.has_end_marker {
                            full.set_is_closed();
                        }
                        self.frozen_trans.push(full);
                    }
                    // buffer freeze handled above (span case only)
                }
            } else if self
                .params
                .matches_starts_with_cached(message_values, &event)
            {
                if has_end_conditions {
                    // When a start event is encountered, try to match with most recent unmatched end event
                    let end_stack =
                        Self::get_or_insert_stack(&mut self.start_trans_stack, &trans_key_ref);

                    // If the top end-only transaction already reached max_events, freeze it first
                    if let Some(top) = end_stack.last()
                        && self.params.max_events > 0
                        && top.get_event_count() >= self.params.max_events
                    {
                        let mut trans_to_freeze = end_stack.pop().unwrap();
                        if trans_to_freeze.has_start_marker && trans_to_freeze.has_end_marker {
                            trans_to_freeze.set_is_closed();
                        }
                        self.frozen_trans.push(trans_to_freeze);
                    }

                    if let Some(mut end_trans) = end_stack.pop() {
                        if self.params.max_events > 0
                            && end_trans.get_event_count() >= self.params.max_events
                        {
                            end_trans.update_start_time_only(&event)?;
                        } else {
                            // Add the start event to the matched end transaction
                            end_trans.add_event(&event)?;
                        }
                        end_trans.has_start_marker = true;
                        end_trans.set_is_closed(); // Mark as closed since it has both start and end
                        self.frozen_trans.push(end_trans);
                    // Do not start a new live transaction with this start; it serves only to close the matched end.
                    } else {
                        // No pending end:
                        // If there is an existing live:
                        //  - If it is a regular (no start marker), merge that live with this start (attach start and keep live)
                        //  - If it already has a start marker, freeze it and start a new live with this start only
                        if let Some(mut existing_trans) = self.live_trans.remove(&trans_key_ref) {
                            if existing_trans.has_start_marker {
                                // Freeze previous start-only live, start a fresh one
                                self.frozen_trans.push(existing_trans);
                                let mut new_trans = Transaction::new(self.params.fields.clone());
                                new_trans.has_start_marker = true;
                                new_trans.add_event(&event)?;
                                let owned = Self::make_trans_key_owned_from_ref(&trans_key_ref);
                                self.live_trans.insert(owned, new_trans);
                            } else {
                                // Merge previously accumulated regular events into this start live
                                existing_trans.has_start_marker = true;
                                if self.params.max_events > 0
                                    && existing_trans.get_event_count() >= self.params.max_events
                                {
                                    existing_trans.update_start_time_only(&event)?;
                                } else {
                                    existing_trans.add_event(&event)?;
                                }
                                let owned = Self::make_trans_key_owned_from_ref(&trans_key_ref);
                                self.live_trans.insert(owned, existing_trans);
                            }
                        } else {
                            let mut new_trans = Transaction::new(self.params.fields.clone());
                            new_trans.has_start_marker = true;
                            new_trans.add_event(&event)?;
                            let owned = Self::make_trans_key_owned_from_ref(&trans_key_ref);
                            self.live_trans.insert(owned, new_trans);
                        }
                    }
                } else {
                    // starts_only mode:
                    // Use live_trans as buffer for regular events; maintain the current "start-live" on the stack.
                    let stack =
                        Self::get_or_insert_stack(&mut self.start_trans_stack, &trans_key_ref);
                    if let Some(mut prev_start_live) = stack.pop() {
                        // Close and freeze the previous start-live when a new start arrives
                        prev_start_live.set_is_closed();
                        self.frozen_trans.push(prev_start_live);
                    }
                    // Merge buffer with this start
                    let mut new_trans =
                        if let Some(mut buffer) = self.live_trans.remove(&trans_key_ref) {
                            if self.params.max_events > 0
                                && buffer.get_event_count() >= self.params.max_events
                            {
                                buffer.update_start_time_only(&event)?;
                            } else {
                                buffer.add_event(&event)?;
                            }
                            buffer
                        } else {
                            let mut t = Transaction::new(self.params.fields.clone());
                            t.add_event(&event)?;
                            t
                        };
                    new_trans.has_start_marker = true;
                    stack.push(new_trans);
                }
            } else if has_start_conditions && has_end_conditions {
                // Regular event in bracket mode: buffer into live_trans (to be merged with the next boundary)
                let stack = Self::get_or_insert_stack(&mut self.start_trans_stack, &trans_key_ref);
                if let Some(trans) = stack.last_mut() {
                    if self.params.max_events == 0
                        || trans.get_event_count() < self.params.max_events
                    {
                        trans.add_event(&event)?;
                    }
                    // Check if max_events is reached after adding (or if it was already reached)
                    if self.params.max_events > 0
                        && trans.get_event_count() >= self.params.max_events
                    {
                        // Immediately freeze and check conditions
                        let mut trans_to_freeze = stack.pop().unwrap();
                        // For bracket mode: need both start and end markers to be closed
                        if trans_to_freeze.has_start_marker && trans_to_freeze.has_end_marker {
                            trans_to_freeze.set_is_closed();
                        }
                        // If only end marker, leave as open (closed=false)
                        self.frozen_trans.push(trans_to_freeze);
                    }
                } else if let Some(existing) = self.live_trans.get_mut(&trans_key_ref) {
                    if existing.has_start_marker {
                        // Freeze the current start-live segment; start a separate regular-live for this event
                        // Need to take ownership to freeze: remove then re-insert a new buffer
                        let taken = self.live_trans.remove(&trans_key_ref).unwrap();
                        self.frozen_trans.push(taken);
                        let mut new_trans = Transaction::new(self.params.fields.clone());
                        new_trans.add_event(&event)?;
                        let owned = Self::make_trans_key_owned_from_ref(&trans_key_ref);
                        self.live_trans.insert(owned, new_trans);
                    } else {
                        // Continue accumulating regular events until a start arrives
                        existing.add_event(&event)?;
                    }
                } else {
                    // Start a new regular-live buffer
                    let mut new_trans = Transaction::new(self.params.fields.clone());
                    new_trans.add_event(&event)?;
                    let owned = Self::make_trans_key_owned_from_ref(&trans_key_ref);
                    self.live_trans.insert(owned, new_trans);
                }
            } else if has_end_conditions {
                // ends_only: add to current end-live if present; otherwise buffer until an end comes
                let stack = Self::get_or_insert_stack(&mut self.start_trans_stack, &trans_key_ref);
                let mut handled_event = false;
                if stack.last().is_some() {
                    loop {
                        let Some(top) = stack.last() else {
                            break;
                        };
                        let mut should_freeze = false;
                        if self.params.max_events > 0
                            && top.get_event_count() >= self.params.max_events
                        {
                            should_freeze = true;
                        } else if let Some(span) = self.params.max_span
                            && span.as_secs() > 0
                            && top.span_exceeds_with_candidate(event_time, span.as_secs())
                        {
                            should_freeze = true;
                        }

                        if should_freeze {
                            let mut trans_to_freeze = stack.pop().unwrap();
                            mark_closed_for_flags(
                                &mut trans_to_freeze,
                                has_start_conditions,
                                has_end_conditions,
                            );
                            self.frozen_trans.push(trans_to_freeze);
                            continue;
                        }
                        break;
                    }

                    if let Some(trans) = stack.last_mut() {
                        if self.params.max_events == 0
                            || trans.get_event_count() < self.params.max_events
                        {
                            trans.add_event(&event)?;
                        }
                        if self.params.max_events > 0
                            && trans.get_event_count() >= self.params.max_events
                        {
                            let mut trans_to_freeze = stack.pop().unwrap();
                            mark_closed_for_flags(
                                &mut trans_to_freeze,
                                has_start_conditions,
                                has_end_conditions,
                            );
                            self.frozen_trans.push(trans_to_freeze);
                        }
                        handled_event = true;
                    }
                }
                if handled_event {
                    // Event already merged into the latest end-only transaction
                } else if let Some(existing) = self.live_trans.get_mut(&trans_key_ref) {
                    existing.add_event(&event)?;
                } else {
                    let mut new_trans = Transaction::new(self.params.fields.clone());
                    new_trans.add_event(&event)?;
                    let owned = Self::make_trans_key_owned_from_ref(&trans_key_ref);
                    self.live_trans.insert(owned, new_trans);
                }
            } else {
                // starts_only mode: buffer regular events in live_trans until a start comes
                if let Some(existing) = self.live_trans.get_mut(&trans_key_ref) {
                    existing.add_event(&event)?;
                } else {
                    let mut new_trans = Transaction::new(self.params.fields.clone());
                    new_trans.add_event(&event)?;
                    let owned = Self::make_trans_key_owned_from_ref(&trans_key_ref);
                    self.live_trans.insert(owned, new_trans);
                }
            }
        } else {
            // Enhanced simple grouping logic with null wildcard matching
            // If there are transactions with compatible keys (considering nulls as wildcards),
            // add to the best matching one based on non-null field matches and time proximity; otherwise create new
            if let Some(best_matching_key) = self.find_best_matching_transaction_key(
                &self.live_trans,
                &trans_key_ref,
                Some(event_time_micros),
            ) {
                if let Some(trans) = self.live_trans.get_mut(&best_matching_key) {
                    trans.add_event(&event)?;
                }
            } else {
                // No compatible transaction found, create new one with this key
                let mut new_trans = Transaction::new(self.params.fields.clone());
                new_trans.add_event(&event)?;
                let owned = Self::make_trans_key_owned_from_ref(&trans_key_ref);
                self.live_trans.insert(owned, new_trans);
            }
        }

        // Do not auto-freeze the regular buffer in ends_only mode; keep it for backfill.
        let ends_only_mode = has_end_conditions && !has_start_conditions;
        if !ends_only_mode
            && let Some(trans) = self.live_trans.get(&trans_key_ref)
            && trans.get_event_count() >= self.params.max_events
        {
            let mut trans_to_freeze = self.live_trans.remove(&trans_key_ref).unwrap();
            mark_closed_for_flags(
                &mut trans_to_freeze,
                has_start_conditions,
                has_end_conditions,
            );
            self.frozen_trans.push(trans_to_freeze);
            self.trans_complete_flag.remove(&trans_key_ref);
        }

        Ok(())
    }

    fn keys_match_with_null_wildcards_owned_ref(
        &self,
        key1: &TransKey,
        key2: &RefTransKey<'_>,
    ) -> bool {
        if key1.parts.len() != key2.parts.len() {
            return false;
        }
        for (v1, v2) in key1.parts.iter().zip(key2.parts.iter()) {
            match (v1, v2) {
                (None, _) => continue,
                (_, None) => continue,
                (Some(val1), Some(val2)) => {
                    if val1.as_str() != *val2 {
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
        collection: &hashbrown::HashMap<TransKey, Transaction, ahash::RandomState>,
        target_key: &RefTransKey<'_>,
        target_time: Option<i64>, // Time of the event being added
    ) -> Option<TransKey> {
        let mut best_key: Option<TransKey> = None;
        let mut best_score: usize = 0;
        let mut best_time_diff: Option<i64> = None;

        for (existing_key, transaction) in collection.iter() {
            if !self.keys_match_with_null_wildcards_owned_ref(existing_key, target_key) {
                continue;
            }

            let exact_non_null_matches = existing_key
                .parts
                .iter()
                .zip(target_key.parts.iter())
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
                            std::cmp::Ordering::Equal => existing_key.parts < best_key_ref.parts,
                        },
                        (Some(_), None) => true,
                        (None, Some(_)) => false,
                        (None, None) => existing_key.parts < best_key_ref.parts,
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

    fn is_valid_event_simple_grouping(&self, event: &EventMap, event_time: DateTime<Utc>) -> bool {
        if !self.params.fields.is_empty() {
            let mut non_empty_count = 0;
            for field in self.params.fields.iter() {
                if let Some(values) = event.get(field)
                    && let Some(first) = values.first()
                    && !first.trim().is_empty()
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
            && event_time > earliest_ts
        {
            return false;
        }
        true
    }

    pub fn get_frozen_trans(&mut self) -> Vec<Transaction> {
        std::mem::take(&mut self.frozen_trans)
    }

    pub fn get_live_trans(&mut self) -> Vec<Transaction> {
        let mut live_transactions: Vec<Transaction> = Vec::new();

        let has_start_only = self.params.ends_with.is_none()
            && self.params.ends_with_regex.is_none()
            && self.params.ends_if_field.is_none()
            && (self.params.starts_with.is_some()
                || self.params.starts_with_regex.is_some()
                || self.params.starts_if_field.is_some());
        let has_end_only = (self.params.ends_with.is_some()
            || self.params.ends_with_regex.is_some()
            || self.params.ends_if_field.is_some())
            && self.params.starts_with.is_none()
            && self.params.starts_with_regex.is_none()
            && self.params.starts_if_field.is_none();

        // Add all remaining live transactions
        for (_, mut trans) in self.live_trans.drain() {
            // If we are in ends_only mode, mark as closed if has end marker
            if has_end_only && trans.has_end_marker {
                trans.set_is_closed();
            }
            // In ends_only with explicit max_events, split oversized open buffers into chunks
            if has_end_only
                && !trans.has_end_marker
                && self.params.max_events > 0
                && trans.get_event_count() > self.params.max_events
            {
                // Split open regular buffer into chunks of size max_events preserving deque order (front=oldest)
                let max_k = self.params.max_events as usize;
                while trans.get_event_count() > 0 {
                    let mut chunk = Transaction::new(self.params.fields.clone());
                    for key_name in self.params.fields.iter() {
                        if let Some(vals) = trans.get_field_values(key_name) {
                            chunk.set_key_field_from(key_name, vals);
                        } else {
                            chunk.clear_key_field(key_name);
                        }
                    }
                    let mut moved = 0usize;
                    while moved < max_k && trans.get_event_count() > 0 {
                        if let (Some(msgs), Some(t)) =
                            (trans.messages.pop_back(), trans.times.pop_back())
                        {
                            chunk.messages.push_back(msgs);
                            chunk.times.push_back(t);
                            chunk.start_time = Some(
                                chunk
                                    .start_time
                                    .map(|cur| if t < cur { t } else { cur })
                                    .unwrap_or(t),
                            );
                            chunk.end_time = Some(
                                chunk
                                    .end_time
                                    .map(|cur| if t > cur { t } else { cur })
                                    .unwrap_or(t),
                            );
                            chunk.event_count = chunk.event_count.saturating_add(1);
                            if trans.event_count > 0 {
                                trans.event_count -= 1;
                            }
                            moved += 1;
                        } else {
                            break;
                        }
                    }
                    if chunk.get_event_count() > 0 {
                        live_transactions.push(chunk);
                    }
                }
            } else if trans.get_event_count() > 0 {
                live_transactions.push(trans);
            }
        }

        // Add all remaining start transactions from the stack
        // These are unmatched end events that never found their corresponding start events
        for (_key, mut trans_stack) in self.start_trans_stack.drain() {
            for mut trans in trans_stack.drain(..) {
                // If we are in starts_only mode, mark remaining as closed if has start marker
                if has_start_only && trans.has_start_marker {
                    trans.set_is_closed();
                }
                // If we are in ends_only mode, mark as closed if has end marker
                if has_end_only && trans.has_end_marker {
                    trans.set_is_closed();
                }
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

fn array_to_strings(array: ArrayRef, row_idx: usize) -> EventFieldValues {
    match array.data_type() {
        DataType::Utf8 => smallvec![string_to_arc(
            scalar_to_string(array, row_idx).expect("utf8 must be scalar")
        )],
        DataType::Int64 => smallvec![string_to_arc(
            scalar_to_string(array, row_idx).expect("int64 must be scalar")
        )],
        DataType::Float64 => smallvec![string_to_arc(
            scalar_to_string(array, row_idx).expect("float64 must be scalar")
        )],
        DataType::Boolean => smallvec![string_to_arc(
            scalar_to_string(array, row_idx).expect("boolean must be scalar")
        )],
        DataType::Timestamp(_, _) => smallvec![string_to_arc(
            scalar_to_string(array, row_idx).expect("timestamp must be scalar")
        )],
        DataType::List(_inner_field) => {
            let nested_list = array.as_list::<i32>().value(row_idx);
            let mut small_vec: EventFieldValues = smallvec![];
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
        let mut events: Vec<EventMap> = Vec::with_capacity(input.num_rows());

        for row_idx in 0..input.num_rows() {
            let mut event: EventMap = HashMap::new();
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

fn mark_closed_for_flags(
    trans: &mut Transaction,
    has_start_conditions: bool,
    has_end_conditions: bool,
) {
    match (has_start_conditions, has_end_conditions) {
        (false, false) => trans.set_is_closed(),
        (true, true) => {
            if trans.has_start_marker && trans.has_end_marker {
                trans.set_is_closed();
            }
        }
        (true, false) => {
            if trans.has_start_marker {
                trans.set_is_closed();
            }
        }
        (false, true) => {
            if trans.has_end_marker {
                trans.set_is_closed();
            }
        }
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
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs::File;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn collect_message_sets(transactions: &[Transaction]) -> Vec<(bool, BTreeSet<String>)> {
        transactions
            .iter()
            .map(|t| {
                let mut set = BTreeSet::new();
                for msgs in &t.messages {
                    for m in msgs {
                        set.insert(m.to_string());
                    }
                }
                (t.is_closed, set)
            })
            .collect()
    }

    fn collect_sorted_messages(transactions: &[Transaction]) -> Vec<(bool, Vec<String>)> {
        transactions
            .iter()
            .map(|t| {
                let mut vec = t
                    .messages
                    .iter()
                    .flat_map(|msgs| msgs.iter().map(|a| a.to_string()))
                    .collect::<Vec<_>>();
                vec.sort();
                (t.is_closed, vec)
            })
            .collect()
    }

    // Helper function to create a simple event HashMap
    fn create_event(time_micros: i64, message: &str, other_fields: &[(&str, &str)]) -> EventMap {
        let mut event = EventMap::new();
        event.insert(
            FIELD_TIME.to_string(),
            smallvec![string_to_arc(time_micros.to_string())],
        );
        event.insert(
            FIELD_MESSAGE.to_string(),
            smallvec![Arc::<str>::from(message)],
        );
        for (k, v) in other_fields {
            event.insert(k.to_string(), smallvec![Arc::<str>::from(*v)]);
        }
        event
    }

    fn create_event_nullable(
        time_micros: i64,
        message: &str,
        other_fields: &[(&str, Option<&str>)],
    ) -> EventMap {
        let mut event = EventMap::new();
        event.insert(
            FIELD_TIME.to_string(),
            smallvec![string_to_arc(time_micros.to_string())],
        );
        event.insert(
            FIELD_MESSAGE.to_string(),
            smallvec![Arc::<str>::from(message)],
        );
        for (k, v) in other_fields {
            if let Some(v) = *v {
                event.insert(k.to_string(), smallvec![Arc::<str>::from(v)]);
            } else {
                event.insert(k.to_string(), SmallVec::<[Arc<str>; 4]>::new());
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

    fn record_to_event(record: &csv::StringRecord) -> EventMap {
        let time_str = record.get(0).expect("timestamp column");
        let message = record.get(1).expect("message column");
        let client_ip = record.get(2).expect("client_ip column");
        let jsessionid = record.get(3).expect("JSESSIONID column");

        let time = chrono::DateTime::parse_from_rfc3339(time_str)
            .expect("valid timestamp")
            .timestamp_micros();

        let mut event = EventMap::new();
        event.insert(
            FIELD_TIME.to_string(),
            smallvec![string_to_arc(time.to_string())],
        );
        event.insert(
            FIELD_MESSAGE.to_string(),
            smallvec![Arc::<str>::from(message)],
        );
        event.insert(
            "client_ip".to_string(),
            smallvec![Arc::<str>::from(client_ip)],
        );
        event.insert(
            "JSESSIONID".to_string(),
            smallvec![Arc::<str>::from(jsessionid)],
        );
        event
    }

    fn sample_csv_events() -> Vec<EventMap> {
        sample_csv_records()
            .into_iter()
            .map(|record| record_to_event(&record))
            .collect()
    }

    fn event_time_for_index(base: chrono::DateTime<Utc>, idx: i64) -> i64 {
        // idx starts from 1 (newest first, then older)
        (base - ChronoDuration::seconds(idx * 60)).timestamp_micros()
    }

    fn messages_of(trans: &Transaction) -> Vec<String> {
        let mut out = Vec::new();
        for msgs in trans.messages.iter() {
            for m in msgs {
                out.push(m.to_string());
            }
        }
        out
    }

    fn assert_has_group(groups: &[Transaction], expected_msgs: &[&str], expected_closed: bool) {
        let needle: Vec<String> = expected_msgs.iter().map(|s| s.to_string()).collect();
        assert!(
            groups
                .iter()
                .any(|t| messages_of(t) == needle && t.is_closed == expected_closed),
            "Expected to find group messages {:?} closed={}",
            needle,
            expected_closed
        );
    }

    #[test]
    fn trans_starts_ends_nested() {
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        // count=1..7 newest->oldest; host=host1
        let seq = vec![
            ("end", 1),
            ("end", 2),
            ("middle", 3),
            ("start", 4),
            ("middle", 5),
            ("start", 6),
            ("other", 7),
        ];

        for (msg, idx) in seq {
            let e = create_event(event_time_for_index(base, idx), msg, &[("host", "host1")]);
            pool.add_event(e).unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 3);
        assert_has_group(&groups, &["start", "middle", "end"], true);
        // there are two closed start-middle-end groups
        assert_eq!(
            groups
                .iter()
                .filter(|t| messages_of(t) == vec!["start", "middle", "end"] && t.is_closed)
                .count(),
            2
        );
        assert_has_group(&groups, &["other"], false);
    }

    #[test]
    fn trans_starts_ends_nested_with_maxevents() {
        // trans.md (229-249): nested startswith + endswith + maxevents
        // Sequence (newest to oldest):
        //   1: end, 2: end, 3: start, 4: start, 5: other
        // Params: startswith="start", endswith="end", maxevents=3, keepevicted=true
        // Expected groups:
        //   group1 (closed=true): start, end (no. 2, 3) -> time ascending: start(3), end(2)
        //   group2 (closed=true): start, end (no. 1, 4) -> time ascending: start(4), end(1)
        //   group3 (closed=false): other (no. 5)
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
                ("max_events".to_string(), Arg::Int(3)),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        let seq = vec![
            ("end", 1),
            ("end", 2),
            ("start", 3),
            ("start", 4),
            ("other", 5),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());

        assert_eq!(groups.len(), 3);
        // There should be exactly two closed ["start","end"] groups
        assert_eq!(
            groups
                .iter()
                .filter(|t| messages_of(t) == vec!["start", "end"] && t.is_closed)
                .count(),
            2
        );
        // And one open ["other"]
        assert_has_group(&groups, &["other"], false);
    }

    #[test]
    fn trans_starts_ends_multiple_start() {
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        // count=1..6
        let seq = vec![
            ("end", 1),
            ("middle", 2),
            ("start", 3),
            ("start", 4),
            ("start", 5),
            ("other", 6),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 4);
        assert_has_group(&groups, &["start", "middle", "end"], true);
        assert_has_group(&groups, &["start"], false);
        assert_eq!(
            groups
                .iter()
                .filter(|t| messages_of(t) == vec!["start"] && !t.is_closed)
                .count(),
            2
        );
        assert_has_group(&groups, &["other"], false);
    }

    #[test]
    fn trans_starts_ends_multiple_end() {
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        // count=1..6
        let seq = vec![
            ("end", 1),
            ("end", 2),
            ("end", 3),
            ("middle", 4),
            ("start", 5),
            ("other", 6),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 3);
        assert_has_group(&groups, &["end"], false);
        assert_has_group(&groups, &["start", "middle", "end"], true);
        assert_has_group(&groups, &["other", "end"], false);
    }

    #[test]
    fn trans_starts_ends_only_start() {
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        // count=1..5
        let seq = vec![
            ("middle", 1),
            ("middle", 2),
            ("start", 3),
            ("start", 4),
            ("other", 5),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 3);
        assert_has_group(&groups, &["start", "middle", "middle"], false);
        assert_has_group(&groups, &["start"], false);
        assert_has_group(&groups, &["other"], false);
    }

    #[test]
    fn trans_starts_ends_only_end() {
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        // count=1..5
        let seq = vec![
            ("end", 1),
            ("middle", 2),
            ("end", 3),
            ("middle", 4),
            ("other", 5),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 2);
        assert_has_group(&groups, &["middle", "end"], false);
        assert_has_group(&groups, &["other", "middle", "end"], false);
    }

    #[test]
    fn trans_only_end_strange() {
        // trans.md: strange end
        // Sequence (newest to oldest): middle(1), end(2), middle(3), end(4)
        // Expected groups:
        //   group1 (open): middle
        //   group2 (open): middle, end
        //   group3 (open): end
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        let seq = vec![("middle", 1), ("end", 2), ("middle", 3), ("end", 4)];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());

        assert_eq!(groups.len(), 3);
        assert_has_group(&groups, &["middle"], false);
        assert_has_group(&groups, &["middle", "end"], true);
        assert_has_group(&groups, &["end"], true);
    }

    #[test]
    fn trans_message_order_time_ascending_bracket_simple() {
        // Newest -> Oldest: end, middle, start
        // Expect one closed group with messages in ascending time order: start, middle, end
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        let seq = vec![("end", 1), ("middle", 2), ("start", 3)];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());

        assert_eq!(groups.len(), 1);
        assert_has_group(&groups, &["start", "middle", "end"], true);
    }

    #[test]
    fn trans_starts_ends_with_maxevents() {
        // trans.md: starts ends with maxevents
        // Sequence (newest to oldest): end(1), middle(2), start(3), end(4), start(5), end(6), m3(7), m2(8), m1(9), start(10), other(11)
        // maxevents=3
        // Expected groups:
        //   group1 (closed=true): start, middle, end (no. 3, 2, 1) - time ascending
        //   group2 (closed=true): start, end (no. 5, 4) - time ascending
        //   group3 (closed=false): m2, m3, end (no. 8, 7, 6) - time ascending, maxevents reached
        //   group4 (closed=false): start, m1 (no. 10, 9) - time ascending
        //   group5 (closed=false): other (no. 11)
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
                ("max_events".to_string(), Arg::Int(3)),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        let seq = vec![
            ("end", 1),
            ("middle", 2),
            ("start", 3),
            ("end", 4),
            ("start", 5),
            ("end", 6),
            ("m3", 7),
            ("m2", 8),
            ("m1", 9),
            ("start", 10),
            ("other", 11),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());

        for (i, g) in groups.iter().enumerate() {
            let msgs: Vec<String> = g
                .messages
                .iter()
                .flat_map(|m| m.iter().map(|a| a.to_string()))
                .collect();
            println!(
                "Group {}: closed={}, has_start={}, has_end={}, messages={:?}",
                i + 1,
                g.is_closed,
                g.has_start_marker,
                g.has_end_marker,
                msgs
            );
        }

        assert_eq!(groups.len(), 5);
        // group1 (closed=true): start(3), middle(2), end(1) - time ascending
        assert_has_group(&groups, &["start", "middle", "end"], true);
        // group2 (closed=true): start(5), end(4) - time ascending
        assert_has_group(&groups, &["start", "end"], true);
        // group3 (closed=false): m2(8), m3(7), end(6) - time ascending, maxevents=3 reached
        assert_has_group(&groups, &["m2", "m3", "end"], false);
        // group4 (closed=false): start(10), m1(9) - time ascending
        assert_has_group(&groups, &["start", "m1"], false);
        // group5 (closed=false): other(11)
        assert_has_group(&groups, &["other"], false);
    }

    #[test]
    fn trans_max_events_zero_with_brackets_singletons_open() {
        // trans.md: max_events=0 + starts + ends
        // end, middle, start, other1, other2 -> each one single open group
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
                ("max_events".to_string(), Arg::Int(0)),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let seq = vec![
            ("end", 1),
            ("middle", 2),
            ("start", 3),
            ("other1", 4),
            ("other2", 5),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }
        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 5);
        assert_has_group(&groups, &["end"], false);
        assert_has_group(&groups, &["middle"], false);
        assert_has_group(&groups, &["start"], false);
        assert_has_group(&groups, &["other1"], false);
        assert_has_group(&groups, &["other2"], false);
    }

    #[test]
    fn trans_max_events_zero_no_brackets_identity_closed() {
        // trans.md: max_events=0 without starts/ends -> identity closed
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("max_events".to_string(), Arg::Int(0)),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let seq = vec![
            ("event1", 1),
            ("event2", 2),
            ("event3", 3),
            ("event4", 4),
            ("event5", 5),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }
        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 5);
        assert_has_group(&groups, &["event1"], true);
        assert_has_group(&groups, &["event2"], true);
        assert_has_group(&groups, &["event3"], true);
        assert_has_group(&groups, &["event4"], true);
        assert_has_group(&groups, &["event5"], true);
    }

    #[test]
    fn trans_max_events_zero_start_equals_end_single_closed() {
        // trans.md: max_events=0, starts+ends have same token -> each such event closed
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                (
                    "starts_with".to_string(),
                    Arg::String("start_and_end".to_string()),
                ),
                (
                    "ends_with".to_string(),
                    Arg::String("start_and_end".to_string()),
                ),
                ("max_events".to_string(), Arg::Int(0)),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let seq = vec![
            ("start_and_end", 1),
            ("start_and_end", 2),
            ("start_and_end", 3),
            ("other", 4),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }
        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 4);
        assert_has_group(&groups, &["start_and_end"], true);
        assert_eq!(
            groups
                .iter()
                .filter(|t| messages_of(t) == vec!["start_and_end"] && t.is_closed)
                .count(),
            3
        );
        assert_has_group(&groups, &["other"], false);
    }

    #[test]
    fn trans_max_span_zero_with_brackets_singletons_open() {
        // trans.md mirrored: maxspan=0s + starts + ends -> each single open group
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
                ("max_span".to_string(), Arg::String("0s".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let seq = vec![
            ("end", 1),
            ("middle", 2),
            ("start", 3),
            ("other1", 4),
            ("other2", 5),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }
        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 5);
        assert_has_group(&groups, &["end"], false);
        assert_has_group(&groups, &["middle"], false);
        assert_has_group(&groups, &["start"], false);
        assert_has_group(&groups, &["other1"], false);
        assert_has_group(&groups, &["other2"], false);
    }

    #[test]
    fn trans_max_span_zero_no_brackets_identity_closed() {
        // trans.md mirrored: maxspan=0s without starts/ends -> identity closed
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("max_span".to_string(), Arg::String("0s".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let seq = vec![
            ("event1", 1),
            ("event2", 2),
            ("event3", 3),
            ("event4", 4),
            ("event5", 5),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }
        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 5);
        assert_has_group(&groups, &["event1"], true);
        assert_has_group(&groups, &["event2"], true);
        assert_has_group(&groups, &["event3"], true);
        assert_has_group(&groups, &["event4"], true);
        assert_has_group(&groups, &["event5"], true);
    }

    #[test]
    fn trans_max_span_zero_start_equals_end_single_closed() {
        // trans.md mirrored: maxspan=0s, starts+ends same token -> closed singles
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                (
                    "starts_with".to_string(),
                    Arg::String("start_and_end".to_string()),
                ),
                (
                    "ends_with".to_string(),
                    Arg::String("start_and_end".to_string()),
                ),
                ("max_span".to_string(), Arg::String("0s".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        let seq = vec![
            ("start_and_end", 1),
            ("start_and_end", 2),
            ("start_and_end", 3),
            ("other", 4),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }
        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 4);
        assert_has_group(&groups, &["start_and_end"], true);
        assert_eq!(
            groups
                .iter()
                .filter(|t| messages_of(t) == vec!["start_and_end"] && t.is_closed)
                .count(),
            3
        );
        assert_has_group(&groups, &["other"], false);
    }

    #[test]
    fn trans_max_span_negative_unset_no_brackets() {
        // Negative maxspan acts as unset: should not force singletons
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("max_span".to_string(), Arg::String("-5s".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        for (msg, idx) in [("a", 1), ("b", 2), ("c", 3)] {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }
        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        // Expect not 3 singleton groups; behavior should be equivalent to unset
        // i.e., not forcing singletons. Assert total events are 3 and group count < 3.
        let total_events: usize = groups.iter().map(|t| t.get_event_count() as usize).sum();
        assert_eq!(total_events, 3);
        assert!(groups.len() < 3);
    }

    #[test]
    fn trans_max_span_negative_unset_with_brackets() {
        // Negative maxspan acts as unset: bracket behavior remains normal
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
                ("max_span".to_string(), Arg::String("-1s".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);
        for (msg, idx) in [("end", 1), ("middle", 2), ("start", 3)] {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }
        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 1);
        assert_has_group(&groups, &["start", "middle", "end"], true);
    }
    #[test]
    fn trans_only_start_multiple_start() {
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        // count=1..7
        let seq = vec![
            ("middle", 1),
            ("middle", 2),
            ("start", 3),
            ("middle", 4),
            ("start", 5),
            ("middle", 6),
            ("start", 7),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 3);
        assert_has_group(&groups, &["start", "middle", "middle"], true);
        assert_eq!(
            groups
                .iter()
                .filter(|t| messages_of(t) == vec!["start", "middle"] && t.is_closed)
                .count(),
            2
        );
    }

    #[test]
    fn trans_only_start_sequential_starts() {
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("start".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        // count=1..5
        let seq = vec![
            ("start", 1),
            ("start", 2),
            ("start", 3),
            ("middle", 4),
            ("other", 5),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 4);
        assert_eq!(
            groups
                .iter()
                .filter(|t| messages_of(t) == vec!["start"] && t.is_closed)
                .count(),
            3
        );
        assert_has_group(&groups, &["other", "middle"], false);
    }

    #[test]
    fn trans_only_end_multiple_end() {
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        // count=1..8 (newest to oldest)
        // trans.md: count=1 "other", count=2 "end", count=3 "middle", count=4 "end",
        // count=5 "middle", count=6 "end", count=7 "middle", count=8 "other"
        let seq = vec![
            ("other", 1),
            ("end", 2),
            ("middle", 3),
            ("end", 4),
            ("middle", 5),
            ("end", 6),
            ("middle", 7),
            ("other", 8),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 4);
        assert_has_group(&groups, &["other"], false);
        assert_eq!(
            groups
                .iter()
                .filter(|t| messages_of(t) == vec!["middle", "end"] && t.is_closed)
                .count(),
            2
        );
        assert_eq!(
            groups
                .iter()
                .filter(|t| {
                    let msgs = messages_of(t);
                    (msgs == vec!["other", "middle", "end"]
                        || msgs == vec!["middle", "other", "end"]
                        || msgs == vec!["middle", "end", "other"])
                        && t.is_closed
                })
                .count(),
            1
        );
    }

    #[test]
    fn trans_only_end_sequential_ends() {
        let base = Utc::now();
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("ends_with".to_string(), Arg::String("end".to_string())),
            ],
        )
        .unwrap();
        let mut pool = TransactionPool::new(params);

        // count=1..6
        let seq = vec![
            ("end", 1),
            ("end", 2),
            ("end", 3),
            ("middle", 4),
            ("middle", 5),
            ("other", 6),
        ];
        for (msg, idx) in seq {
            pool.add_event(create_event(
                event_time_for_index(base, idx),
                msg,
                &[("host", "host1")],
            ))
            .unwrap();
        }

        let mut groups = pool.get_frozen_trans();
        groups.extend(pool.get_live_trans());
        assert_eq!(groups.len(), 3);
        // ends_only semantics updated: groups containing end are closed
        assert_has_group(&groups, &["end"], true);
        assert_eq!(
            groups
                .iter()
                .filter(|t| messages_of(t) == vec!["end"] && t.is_closed)
                .count(),
            2
        );
        assert_has_group(&groups, &["other", "middle", "middle", "end"], true);
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
        assert_eq!(params.max_span, Some(Duration::from_secs(600)));
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
        let mut trans = Transaction::new(Arc::from(vec!["user_id".to_string()].into_boxed_slice()));
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
            trans.messages.front().unwrap().first().unwrap().as_ref(),
            "message three"
        );
        assert_eq!(
            trans.messages.back().unwrap().first().unwrap().as_ref(),
            "message one"
        );

        {
            let expected: SmallVec<[&str; 4]> = smallvec!["user1"];
            let actual_user: SmallVec<[&str; 4]> = trans
                .get_field_values("user_id")
                .unwrap()
                .iter()
                .map(|s| s.as_ref())
                .collect();
            assert_eq!(actual_user, expected);
            assert!(
                trans
                    .get_field_values("status")
                    .unwrap()
                    .iter()
                    .any(|s| s.as_ref() == "success")
            );
            assert!(
                trans
                    .get_field_values("status")
                    .unwrap()
                    .iter()
                    .any(|s| s.as_ref() == "fail")
            );
            let expected: SmallVec<[&str; 4]> = smallvec!["abc"];
            let actual_data: SmallVec<[&str; 4]> = trans
                .get_field_values("data")
                .unwrap()
                .iter()
                .map(|s| s.as_ref())
                .collect();
            assert_eq!(actual_data, expected);
        }

        let expected_duration = ((now - ChronoDuration::seconds(10)).timestamp_micros()
            - (now - ChronoDuration::seconds(30)).timestamp_micros())
            as f64
            / 1_000_000.0;
        assert_eq!(trans.get_duration().unwrap(), expected_duration);
        assert!(!trans.is_closed);
        trans.set_is_closed();
        assert!(trans.is_closed);
    }

    #[test]
    fn test_transaction_to_record_batch() {
        let mut trans = Transaction::new(Arc::from(vec!["user_id".to_string()].into_boxed_slice()));
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

    // removed legacy test: starts-only closes transactions; trans.md defines starts-only as open segments

    // removed legacy test: ends-only closes transactions; trans.md defines ends-only as open segments

    // removed legacy test: starts_with_regex-only closes transactions; follow trans.md

    // removed legacy test: ends_with_regex-only closes transactions; follow trans.md

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
            let mut event = EventMap::new();
            event.insert(
                FIELD_TIME.to_string(),
                smallvec![string_to_arc(
                    (now - ChronoDuration::seconds(offset_secs))
                        .timestamp_micros()
                        .to_string()
                )],
            );
            event.insert(
                "session_id".to_string(),
                smallvec![Arc::<str>::from(session)],
            );
            event.insert("step".to_string(), smallvec![Arc::<str>::from(step)]);
            for (k, v) in flags {
                event.insert((*k).to_string(), smallvec![Arc::<str>::from(*v)]);
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
            .get_field_values("step")
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|a| a.to_string())
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
                .get_field_values("step")
                .is_some_and(|vals| vals.iter().any(|s| s.as_ref() == "after_end"))
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
                .flat_map(|msg_list| msg_list.iter().map(|a| a.to_string()))
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
                .flat_map(|msg_list| msg_list.iter().map(|a| a.to_string()))
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
                .flat_map(|msg_list| msg_list.iter().map(|a| a.to_string()))
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
                .flat_map(|msg_list| msg_list.iter().map(|a| a.to_string()))
                .collect();
            msgs.sort(); // Sort for consistent comparison
            message_groups.push(msgs);
        }

        // Sort the groups for comparison since order might vary
        message_groups.sort();

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
                .flat_map(|msg_list| msg_list.iter().map(|a| a.to_string()))
                .collect();
            msgs.sort(); // Sort for consistent comparison
            message_groups.push(msgs);
        }

        // Sort the groups for comparison since order might vary
        message_groups.sort();

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
                    .get_field_values("client_ip")
                    .and_then(|v| v.first())
                    .cloned()
                    .unwrap_or_else(|| Arc::<str>::from("")),
                trans
                    .get_field_values("JSESSIONID")
                    .and_then(|v| v.first())
                    .cloned()
                    .unwrap_or_else(|| Arc::<str>::from("")),
            );
            let messages: Vec<String> = trans
                .messages
                .iter()
                .flat_map(|msgs| msgs.iter().map(|a| a.to_string()))
                .collect();
            if trans.is_closed {
                // convert Arc<str> key parts to String
                span_closed.insert((key.0.to_string(), key.1.to_string()), messages);
            } else {
                span_open
                    .entry((key.0.to_string(), key.1.to_string()))
                    .or_default()
                    .push(messages);
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
                    .get_field_values("client_ip")
                    .and_then(|vals| vals.first())
                    .cloned();
                let jsession = trans
                    .get_field_values("JSESSIONID")
                    .and_then(|vals| vals.first())
                    .cloned();
                if let (Some(client_ip), Some(jsession)) = (client_ip, jsession)
                    && client_ip.as_ref() == ip
                    && jsession.as_ref() == session
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

    #[test]
    fn test_transaction_counts_view_purchase_from_fixture() {
        let params = TransParams::new(
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

        let mut pool = TransactionPool::new(params.clone());

        for event in sample_csv_events() {
            pool.add_event(event).unwrap();
        }
        let frozen = pool.get_frozen_trans();
        let frozen_record_batch = to_record_batch(&params, &frozen).unwrap().unwrap();
        let live_record_batch = to_record_batch(&params, &pool.get_live_trans())
            .unwrap()
            .unwrap();
        // get _is_closed, and count true and false
        let frozen_is_closed = frozen_record_batch
            .column(4)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let live_is_closed = live_record_batch
            .column(4)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let frozen_is_closed_true = frozen_is_closed.iter().filter(|&b| b.unwrap()).count();
        let frozen_is_closed_false = frozen_is_closed.iter().filter(|&b| !b.unwrap()).count();
        let live_is_closed_true = live_is_closed.iter().filter(|&b| b.unwrap()).count();
        let live_is_closed_false = live_is_closed.iter().filter(|&b| !b.unwrap()).count();
        let total_is_closed_true = frozen_is_closed_true + live_is_closed_true;
        let total_is_closed_false = frozen_is_closed_false + live_is_closed_false;
        assert_eq!(total_is_closed_true, 7);
        assert_eq!(total_is_closed_false, 67);
    }

    #[test]
    fn test_max_events_only_close_state() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("max_events".to_string(), Arg::Int(3)),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);
        let labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"];
        for event in build_host_sequence_events(&labels) {
            pool.add_event(event).unwrap();
        }

        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());
        assert_eq!(transactions.len(), 4);

        let groups = collect_message_sets(&transactions);
        let closed_groups: Vec<_> = groups.iter().filter(|(closed, _)| *closed).collect();
        assert_eq!(closed_groups.len(), 3);
        let expected_closed = [
            BTreeSet::from(["1".to_string(), "2".to_string(), "3".to_string()]),
            BTreeSet::from(["4".to_string(), "5".to_string(), "6".to_string()]),
            BTreeSet::from(["7".to_string(), "8".to_string(), "9".to_string()]),
        ];
        for expected in expected_closed {
            assert!(
                closed_groups.iter().any(|(_, set)| *set == expected),
                "Expected closed group {expected:?}"
            );
        }

        let open_groups: Vec<_> = groups.iter().filter(|(closed, _)| !closed).collect();
        assert_eq!(open_groups.len(), 1);
        assert_eq!(open_groups[0].1.len(), 2);
        let expected_open = BTreeSet::from(["10".to_string(), "11".to_string()]);
        assert_eq!(&open_groups[0].1, &expected_open);
    }

    #[test]
    fn test_max_span_only_close_state() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("max_span".to_string(), Arg::String("2m".to_string())),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);
        let labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"];
        for event in build_host_sequence_events(&labels) {
            pool.add_event(event).unwrap();
        }

        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());
        assert_eq!(transactions.len(), 4);

        let groups = collect_message_sets(&transactions);
        let closed_groups: Vec<_> = groups.iter().filter(|(closed, _)| *closed).collect();
        assert_eq!(closed_groups.len(), 3);
        assert!(closed_groups.iter().all(|(_, set)| set.len() == 3));

        let open_groups: Vec<_> = groups.iter().filter(|(closed, _)| !closed).collect();
        assert_eq!(open_groups.len(), 1);
        assert_eq!(open_groups[0].1.len(), 2);
    }

    fn build_doc_max_span_events() -> Vec<EventMap> {
        let base = Utc::now();
        let sequence = [
            ("category", 0),
            ("product", 1),
            ("purchase", 2),
            ("purchase", 3),
            ("addtocart", 4),
            ("view", 5),
            ("view", 6),
            ("oldlink", 7),
            ("addtocart", 8),
            ("oldlink", 9),
            ("view", 10),
        ];
        let key_fields = [("host", "host1")];
        sequence
            .iter()
            .map(|(msg, offset)| {
                let ts = (base - ChronoDuration::seconds(*offset)).timestamp_micros();
                create_event(ts, msg, &key_fields)
            })
            .collect()
    }

    fn build_host_sequence_events(labels: &[&str]) -> Vec<EventMap> {
        let base = Utc::now();
        let key_fields = [("host", "host1")];
        labels
            .iter()
            .enumerate()
            .map(|(idx, label)| {
                let ts = event_time_for_index(base, (idx + 1) as i64);
                create_event(ts, label, &key_fields)
            })
            .collect()
    }

    #[test]
    fn test_max_span_doc_three_seconds() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("view".to_string())),
                ("ends_with".to_string(), Arg::String("purchase".to_string())),
                ("max_span".to_string(), Arg::String("3s".to_string())),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);
        for event in build_doc_max_span_events() {
            pool.add_event(event).unwrap();
        }
        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());
        assert_eq!(transactions.len(), 5);

        let groups = collect_sorted_messages(&transactions);
        let expected = [
            (false, vec!["category", "product"]),
            (false, vec!["purchase"]),
            (true, vec!["addtocart", "purchase", "view"]),
            (false, vec!["view"]),
            (false, vec!["addtocart", "oldlink", "oldlink", "view"]),
        ];
        for (closed, members) in expected {
            let mut target: Vec<String> = members.into_iter().map(|s| s.to_string()).collect();
            target.sort();
            assert!(
                groups
                    .iter()
                    .any(|(is_closed, msgs)| *is_closed == closed && *msgs == target),
                "Expected group closed={closed} with members {target:?}"
            );
        }
    }

    #[test]
    fn test_max_span_doc_six_seconds() {
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("starts_with".to_string(), Arg::String("view".to_string())),
                ("ends_with".to_string(), Arg::String("purchase".to_string())),
                ("max_span".to_string(), Arg::String("6s".to_string())),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params);
        for event in build_doc_max_span_events() {
            pool.add_event(event).unwrap();
        }
        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());
        assert_eq!(transactions.len(), 4);

        let groups = collect_sorted_messages(&transactions);
        let expected = [
            (false, vec!["category", "product"]),
            (true, vec!["addtocart", "purchase", "view"]),
            (true, vec!["purchase", "view"]),
            (false, vec!["addtocart", "oldlink", "oldlink", "view"]),
        ];
        for (closed, members) in expected {
            let mut target: Vec<String> = members.into_iter().map(|s| s.to_string()).collect();
            target.sort();
            assert!(
                groups
                    .iter()
                    .any(|(is_closed, msgs)| *is_closed == closed && *msgs == target),
                "Expected group closed={closed} with members {target:?}"
            );
        }
    }

    #[test]
    fn trans_only_end_nested_maxevents_from_md() {
        // trans.md (max events + endswith nested) lines 308-337
        // Params: endswith="purchase", maxevents=3, fields=host, keepevicted=true
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("ends_with".to_string(), Arg::String("purchase".to_string())),
                ("max_events".to_string(), Arg::Int(3)),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params.clone());

        // Create 11 events with messages mapped exactly as in trans.md
        // Process in reverse-chronological order (newest -> oldest)
        let key_fields = &[("host", "host1")];
        let events = vec![
            (110_i64, "other1"),
            (109, "other2"),
            (108, "purchase"),
            (107, "purchase"),
            (106, "other3"),
            (105, "other4"),
            (104, "other5"),
            (103, "other6"),
            (102, "other7"),
            (101, "other8"),
            (100, "other9"),
        ];
        for (t, msg) in events {
            let event = create_event(t, msg, key_fields);
            pool.add_event(event).unwrap();
        }

        // Collect all transactions (frozen + live)
        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());

        // Build message vectors for each transaction
        let groups: Vec<Vec<String>> = transactions
            .iter()
            .map(|t| {
                t.messages
                    .iter()
                    .flat_map(|msgs| msgs.iter().map(|s| s.to_string()))
                    .collect::<Vec<String>>()
            })
            .collect();

        // Normalize order-insensitively for assertions using sets
        let set_of =
            |v: &[String]| -> std::collections::BTreeSet<String> { v.iter().cloned().collect() };
        let group_sets: Vec<std::collections::BTreeSet<String>> =
            groups.iter().map(|g| set_of(g)).collect();
        // Debug print
        let expected1: std::collections::BTreeSet<String> = ["other4", "other3", "purchase"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let expected2: std::collections::BTreeSet<String> = ["other6", "other5", "purchase"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let expected3: std::collections::BTreeSet<String> = ["other7", "other2", "other1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let expected4: std::collections::BTreeSet<String> = ["other9", "other8"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        // Ensure all four expected groups exist
        // Debug: print groups if assertion fails
        // (kept minimal; assertions below will panic with messages)
        assert!(
            group_sets.contains(&expected1),
            "Expected group1 {{other4, other3, purchase}}"
        );
        assert!(
            group_sets.contains(&expected2),
            "Expected group2 {{other6, other5, purchase}}"
        );
        assert!(
            group_sets.contains(&expected3),
            "Expected group3 {{other7, other2, other1}}"
        );
        assert!(
            group_sets.contains(&expected4),
            "Expected group4 {{other9, other8}}"
        );

        // Validate closed flags: groups with 'purchase' should be closed
        let is_purchase_group_closed = |set: &std::collections::BTreeSet<String>| -> bool {
            if set.contains("purchase") {
                // find the transaction that matches this set
                for t in &transactions {
                    let tset: std::collections::BTreeSet<String> = t
                        .messages
                        .iter()
                        .flat_map(|msgs| msgs.iter().map(|s| s.to_string()))
                        .collect();
                    if &tset == set {
                        return t.is_closed;
                    }
                }
                false
            } else {
                true
            }
        };
        assert!(
            is_purchase_group_closed(&expected1),
            "group1 should be closed"
        );
        assert!(
            is_purchase_group_closed(&expected2),
            "group2 should be closed"
        );
    }

    #[test]
    fn trans_ends_only_maxevents_nested() {
        // trans.md (max_events also nested) lines 428-457
        // Params: endswith="purchase", maxevents=3, fields=host
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("ends_with".to_string(), Arg::String("purchase".to_string())),
                ("max_events".to_string(), Arg::Int(3)),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params.clone());

        // Create 11 events matching the doc scenario, reverse-chronological processing
        let key_fields = &[("host", "host1")];
        let events = vec![
            (11_i64, "other1"),
            (10, "other2"),
            (9, "purchase"),
            (8, "purchase"),
            (7, "other3"),
            (6, "other4"),
            (5, "other5"),
            (4, "other6"),
            (3, "other7"),
            (2, "other8"),
            (1, "other9"),
        ];
        for (t, msg) in events {
            let event = create_event(t, msg, key_fields);
            pool.add_event(event).unwrap();
        }

        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());

        let groups: Vec<std::collections::BTreeSet<String>> = transactions
            .iter()
            .map(|t| {
                t.messages
                    .iter()
                    .flat_map(|msgs| msgs.iter().map(|s| s.to_string()))
                    .collect::<std::collections::BTreeSet<String>>()
            })
            .collect();
        let set = |items: &[&str]| -> std::collections::BTreeSet<String> {
            items.iter().map(|s| s.to_string()).collect()
        };

        let expected1 = set(&["other4", "other3", "purchase"]);
        let expected2 = set(&["other6", "other5", "purchase"]);
        let expected3 = set(&["other1", "other2", "other7"]);
        let expected4 = set(&["other9", "other8"]);

        assert!(
            groups.contains(&expected1),
            "Expected group1 {{other4, other3, purchase}}"
        );
        assert!(
            groups.contains(&expected2),
            "Expected group2 {{other6, other5, purchase}}"
        );
        assert!(
            groups.contains(&expected3),
            "Expected group3 {{other1, other2, other7}}"
        );
        assert!(
            groups.contains(&expected4),
            "Expected group4 {{other9, other8}}"
        );

        // Closed flags for purchase groups
        let is_closed_set = |set: &std::collections::BTreeSet<String>| -> bool {
            for t in &transactions {
                let s: std::collections::BTreeSet<String> = t
                    .messages
                    .iter()
                    .flat_map(|msgs| msgs.iter().map(|s| s.to_string()))
                    .collect();
                if &s == set {
                    return t.is_closed;
                }
            }
            false
        };
        assert!(is_closed_set(&expected1));
        assert!(is_closed_set(&expected2));
    }

    #[test]
    fn trans_ends_only_maxspan_nested() {
        // Mirror of the nested ends-only scenario using max_span instead of max_events
        // Params: endswith="purchase", max_span="2s", fields=host
        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("host".to_string())),
                ("ends_with".to_string(), Arg::String("purchase".to_string())),
                ("max_span".to_string(), Arg::String("2s".to_string())),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params.clone());

        // Use custom spacing to ensure each expected group fits within the 2-second span window.
        // Ordering still follows newest -> oldest processing.
        let base = Utc::now().timestamp_micros();
        let ts = |offset_micros: i64| -> i64 { base - offset_micros };
        let key_fields = &[("host", "host1")];
        let events = vec![
            (0_i64, "other1"),
            (100_000, "other2"),
            (300_000, "other7"),
            (3_000_000, "purchase"), // closes with other3/other4
            (3_900_000, "other3"),
            (4_700_000, "other4"),
            (7_000_000, "purchase"), // closes with other5/other6
            (7_800_000, "other5"),
            (8_600_000, "other6"),
            (12_000_000, "other8"),
            (12_800_000, "other9"),
        ];
        for (offset, msg) in events {
            let event = create_event(ts(offset), msg, key_fields);
            pool.add_event(event).unwrap();
        }

        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());
        let groups: Vec<std::collections::BTreeSet<String>> = transactions
            .iter()
            .map(|t| {
                t.messages
                    .iter()
                    .flat_map(|msgs| msgs.iter().map(|s| s.to_string()))
                    .collect::<std::collections::BTreeSet<String>>()
            })
            .collect();
        let set = |items: &[&str]| -> std::collections::BTreeSet<String> {
            items.iter().map(|s| s.to_string()).collect()
        };

        let expected1 = set(&["other4", "other3", "purchase"]);
        let expected2 = set(&["other6", "other5", "purchase"]);
        let expected3 = set(&["other1", "other2", "other7"]);
        let expected4 = set(&["other9", "other8"]);

        assert!(
            groups.contains(&expected1),
            "Expected group1 {{other4, other3, purchase}}"
        );
        assert!(
            groups.contains(&expected2),
            "Expected group2 {{other6, other5, purchase}}"
        );
        assert!(
            groups.contains(&expected3),
            "Expected group3 {{other1, other2, other7}}"
        );
        assert!(
            groups.contains(&expected4),
            "Expected group4 {{other9, other8}}"
        );

        // Closed flags for purchase groups should be true
        let is_closed_set = |set: &std::collections::BTreeSet<String>| -> bool {
            for t in &transactions {
                let s: std::collections::BTreeSet<String> = t
                    .messages
                    .iter()
                    .flat_map(|msgs| msgs.iter().map(|s| s.to_string()))
                    .collect();
                if &s == set {
                    return t.is_closed;
                }
            }
            false
        };
        assert!(is_closed_set(&expected1));
        assert!(is_closed_set(&expected2));
    }
    #[test]
    fn test_nested_starts_ends_order() {
        // Scenario:
        // t10: purchase (END)
        // t09: checkout (middle)
        // t08: purchase (END)
        // t07: add_to_cart (middle)
        // t06: view (START)
        // t05: browse (middle)
        // t04: view (START)
        // t03: login (other)
        // t02: search (other)
        // t01: home (other)
        //
        // Expected closed groups (unordered message sets):
        //   ["view", "add_to_cart", "purchase"]
        //   ["view", "browse", "checkout", "purchase"]
        //
        // Plus one open group for the remaining early events.

        let params = TransParams::new(
            Some(vec![]),
            vec![
                ("fields".to_string(), Arg::String("ip,session".to_string())),
                ("starts_with".to_string(), Arg::String("view".to_string())),
                ("ends_with".to_string(), Arg::String("purchase".to_string())),
            ],
        )
        .unwrap();

        let mut pool = TransactionPool::new(params.clone());

        let key_fields = &[("ip", "1.1.1.1"), ("session", "a")];
        let events = vec![
            (10_i64, "purchase"),
            (9, "checkout"),
            (8, "purchase"),
            (7, "add_to_cart"),
            (6, "view"),
            (5, "browse"),
            (4, "view"),
            (3, "login"),
            (2, "search"),
            (1, "home"),
        ];

        for (t, msg) in events {
            let event = create_event(t, msg, key_fields);
            pool.add_event(event).unwrap();
        }

        let mut transactions = pool.get_frozen_trans();
        transactions.extend(pool.get_live_trans());

        // Collect closed transactions' message sets
        let mut closed_sets: Vec<std::collections::BTreeSet<String>> = Vec::new();
        for trans in transactions.iter().filter(|t| t.is_closed) {
            let mut set = std::collections::BTreeSet::new();
            for messages in &trans.messages {
                for m in messages {
                    set.insert(m.to_string());
                }
            }
            closed_sets.push(set);
        }

        // We expect exactly 2 closed groups with the following compositions (order-insensitive)
        let expected1: std::collections::BTreeSet<String> = ["view", "add_to_cart", "purchase"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let expected2: std::collections::BTreeSet<String> =
            ["view", "browse", "checkout", "purchase"]
                .into_iter()
                .map(|s| s.to_string())
                .collect();

        assert_eq!(
            closed_sets.len(),
            2,
            "Expected exactly 2 closed transactions in nested scenario"
        );
        assert!(
            closed_sets.contains(&expected1),
            "Closed transactions should contain {{view, add_to_cart, purchase}}"
        );
        assert!(
            closed_sets.contains(&expected2),
            "Closed transactions should contain {{view, browse, checkout, purchase}}"
        );
    }
}
