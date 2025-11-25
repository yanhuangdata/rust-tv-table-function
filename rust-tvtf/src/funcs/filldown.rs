use anyhow::Context;
use arrow::{
    array::{Array, ArrayRef, RecordBatch, UInt32Array},
    compute::{concat, take},
};
use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::DataType;

use crate::TableFunction;
use rust_tvtf_api::arg::{Arg, Args};

#[derive(Debug)]
pub struct Filldown {
    column_names: Vec<String>,
    last_values: HashMap<String, Option<ArrayRef>>,
}
impl Filldown {
    pub fn new(params: Option<Args>) -> anyhow::Result<Filldown> {
        let column_names = if let Some(params) = params {
            let scalars = params
                .into_iter()
                .filter(|p| p.is_scalar())
                .collect::<Vec<_>>();

            if !scalars.is_empty() {
                let arg0 = scalars
                    .first()
                    .context("No column names parameter provided")?;

                let Arg::String(column_names_str) = arg0 else {
                    return Err(anyhow::anyhow!("Column names must be a string"));
                };

                // Parse comma-separated column names
                column_names_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<String>>()
            } else {
                Vec::new()
            }
        } else {
            return Err(anyhow::anyhow!("Column names parameter is required"));
        };

        Ok(Filldown {
            column_names,
            last_values: HashMap::new(),
        })
    }

    fn need_filldown(&self, col_name: &str) -> bool {
        if self.column_names.is_empty() {
            return true;
        }
        for name in &self.column_names {
            if name == col_name {
                return true;
            }
            if name.ends_with('*') {
                let prefix = name.trim_end_matches('*');
                if col_name.starts_with(prefix) {
                    return true;
                }
            }
        }
        false
    }
}

fn extract_last_value(col: &ArrayRef) -> anyhow::Result<Option<ArrayRef>> {
    let n = col.len();
    if n == 0 {
        return Ok(None);
    }
    for i in (0..n).rev() {
        if col.is_valid(i) {
            let single = col.slice(i, 1);
            return Ok(Some(single));
        }
    }
    Ok(None)
}

impl TableFunction for Filldown {
    fn process(&mut self, input: RecordBatch) -> anyhow::Result<Option<RecordBatch>> {
        // create a new RecordBatch with same schema as input,
        // replacing nulls with last non-null value for each column
        let schema = input.schema();
        let num_rows = input.num_rows();
        let mut output_columns: Vec<ArrayRef> = Vec::with_capacity(input.num_columns());

        for col_idx in 0..input.num_columns() {
            let col = input.column(col_idx).clone();
            let col_name = schema.field(col_idx).name();

            // No nulls or column not specified for filldown; copy as is
            if !self.need_filldown(col_name) {
                output_columns.push(col);
                continue;
            }

            if col.null_count() == 0 {
                self.last_values
                    .insert(col_name.to_string(), extract_last_value(&col)?);
                output_columns.push(col);
                continue;
            }

            // If column type is `Null`, just produce a null array of same length
            if schema.field(col_idx).data_type() == &DataType::Null {
                output_columns.push(Arc::new(arrow::array::new_null_array(
                    &DataType::Null,
                    num_rows,
                )));
                self.last_values.insert(col_name.to_string(), None);
                continue;
            }

            // Build source: optionally prepend saved last value
            let (source_array, offset) = if let Some(Some(prev)) = self.last_values.get(col_name) {
                // concat requires &[&dyn Array]
                let refs: Vec<&dyn Array> = vec![prev.as_ref(), col.as_ref()];
                let concatenated = concat(&refs).context("Failed to concat arrays for filldown")?;
                (concatenated, 1u32)
            } else {
                (col.clone(), 0u32)
            };

            // Build indices that refer to `source_array`. When prev existed, indices are shifted by `offset`.
            let mut last_idx: Option<u32> = None;
            if offset == 1 {
                // if prev exists and is valid at position 0
                if source_array.is_valid(0) {
                    last_idx = Some(0);
                }
            }

            let mut indices: Vec<Option<u32>> = Vec::with_capacity(num_rows);
            for row in 0..num_rows {
                if col.is_valid(row) {
                    let idx = (row as u32) + offset;
                    last_idx = Some(idx);
                    indices.push(Some(idx));
                } else if let Some(li) = last_idx {
                    indices.push(Some(li));
                } else {
                    indices.push(None);
                }
            }

            let indices_array = Arc::new(UInt32Array::from(indices)) as ArrayRef;

            let taken = take(source_array.as_ref(), indices_array.as_ref(), None)
                .context("Failed to take values for filldown across batches")?;

            // Update saved last value: if last_idx exists, extract that single element from `source_array`
            if let Some(li) = last_idx {
                let idx_arr = UInt32Array::from(vec![li]);
                let idx_ref = Arc::new(idx_arr) as ArrayRef;
                let last_single = take(source_array.as_ref(), idx_ref.as_ref(), None)
                    .context("Failed to take final last_value for filldown")?;
                self.last_values
                    .insert(col_name.to_string(), Some(last_single));
            } else {
                // no non-null seen; keep previous (or set None) - here set None
                self.last_values.insert(col_name.to_string(), None);
            }

            output_columns.push(taken);
        }

        let out_batch = RecordBatch::try_new(schema.clone(), output_columns)
            .context("Failed to create output RecordBatch for filldown")?;

        Ok(Some(out_batch))
    }
    fn finalize(&mut self) -> anyhow::Result<Option<RecordBatch>> {
        self.last_values.clear();
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::ArrayRef;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use rust_tvtf_api::TableFunction;
    use std::sync::Arc;

    #[test]
    fn test_filldown_basic() {
        // create input RecordBatch with some nulls

        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef;
        let name_array = Arc::new(StringArray::from(vec![
            Some("Alice"),
            None,
            Some("Bob"),
            None,
            None,
        ])) as ArrayRef;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        let input_batch =
            RecordBatch::try_new(schema.clone(), vec![id_array.clone(), name_array.clone()])
                .expect("Failed to create input RecordBatch");

        // create Filldown table function
        let params = Args::from(vec![Arg::String("".to_string())]);
        let mut filldown = super::Filldown::new(Some(params)).expect("Failed to create Filldown");
        // process input batch
        let output_batch = filldown
            .process(input_batch)
            .expect("Failed to process batch")
            .expect("No output batch");

        // verify output
        let expected_name_array = Arc::new(StringArray::from(vec![
            Some("Alice"),
            Some("Alice"),
            Some("Bob"),
            Some("Bob"),
            Some("Bob"),
        ])) as ArrayRef;
        assert_eq!(output_batch.column(0).as_ref(), id_array.as_ref());
        assert_eq!(
            output_batch.column(1).as_ref(),
            expected_name_array.as_ref()
        );
    }

    #[test]
    fn test_filldown_args() {
        // create input RecordBatch with some nulls

        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef;
        let name_array = Arc::new(StringArray::from(vec![
            Some("Alice"),
            None,
            Some("Bob"),
            None,
            None,
        ])) as ArrayRef;

        let city_array = Arc::new(StringArray::from(vec![
            Some("NY"),
            None,
            None,
            Some("LA"),
            None,
        ])) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("city", DataType::Utf8, true),
        ]));

        let input_batch = RecordBatch::try_new(
            schema.clone(),
            vec![id_array.clone(), name_array.clone(), city_array.clone()],
        )
        .expect("Failed to create input RecordBatch");

        // create Filldown table function
        let params = Args::from(vec![Arg::String("city".to_string())]);
        let mut filldown = super::Filldown::new(Some(params)).expect("Failed to create Filldown");
        // process input batch
        let output_batch = filldown
            .process(input_batch)
            .expect("Failed to process batch")
            .expect("No output batch");

        // verify output
        let expected_city_array = Arc::new(StringArray::from(vec![
            Some("NY"),
            Some("NY"),
            Some("NY"),
            Some("LA"),
            Some("LA"),
        ])) as ArrayRef;
        assert_eq!(output_batch.column(0).as_ref(), id_array.as_ref());
        assert_eq!(output_batch.column(1).as_ref(), name_array.as_ref());
        assert_eq!(
            output_batch.column(2).as_ref(),
            expected_city_array.as_ref()
        );
    }

    #[test]
    fn test_filldown_args_regex() {
        // create input RecordBatch with some nulls

        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef;
        let name_array = Arc::new(StringArray::from(vec![
            Some("Alice"),
            None,
            Some("Bob"),
            None,
            None,
        ])) as ArrayRef;

        let city_array = Arc::new(StringArray::from(vec![
            Some("NY"),
            None,
            None,
            Some("LA"),
            None,
        ])) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("city", DataType::Utf8, true),
        ]));

        let input_batch = RecordBatch::try_new(
            schema.clone(),
            vec![id_array.clone(), name_array.clone(), city_array.clone()],
        )
        .expect("Failed to create input RecordBatch");

        // create Filldown table function
        let params = Args::from(vec![Arg::String("c*".to_string())]);
        let mut filldown = super::Filldown::new(Some(params)).expect("Failed to create Filldown");
        // process input batch
        let output_batch = filldown
            .process(input_batch)
            .expect("Failed to process batch")
            .expect("No output batch");

        // verify output
        let expected_city_array = Arc::new(StringArray::from(vec![
            Some("NY"),
            Some("NY"),
            Some("NY"),
            Some("LA"),
            Some("LA"),
        ])) as ArrayRef;
        assert_eq!(output_batch.column(0).as_ref(), id_array.as_ref());
        assert_eq!(output_batch.column(1).as_ref(), name_array.as_ref());
        assert_eq!(
            output_batch.column(2).as_ref(),
            expected_city_array.as_ref()
        );
    }

    #[test]
    fn test_filldown_multi_batch() {
        // create input RecordBatches with some nulls
        let id_array1 = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let name_array1 =
            Arc::new(StringArray::from(vec![Some("Alice"), None, Some("Bob")])) as ArrayRef;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        let input_batch1 =
            RecordBatch::try_new(schema.clone(), vec![id_array1.clone(), name_array1.clone()])
                .expect("Failed to create input RecordBatch 1");

        let id_array2 = Arc::new(Int32Array::from(vec![4, 5])) as ArrayRef;
        let name_array2 = Arc::new(StringArray::from(vec![None::<&str>, None])) as ArrayRef;

        let input_batch2 =
            RecordBatch::try_new(schema.clone(), vec![id_array2.clone(), name_array2.clone()])
                .expect("Failed to create input RecordBatch 2");

        // create Filldown table function
        let params = Args::from(vec![Arg::String("".to_string())]);
        let mut filldown = super::Filldown::new(Some(params)).expect("Failed to create Filldown");

        // process first input batch
        let output_batch1 = filldown
            .process(input_batch1)
            .expect("Failed to process batch 1")
            .expect("No output batch 1");

        // process second input batch
        let output_batch2 = filldown
            .process(input_batch2)
            .expect("Failed to process batch 2")
            .expect("No output batch 2");

        // verify output of first batch
        let expected_name_array1 = Arc::new(StringArray::from(vec![
            Some("Alice"),
            Some("Alice"),
            Some("Bob"),
        ])) as ArrayRef;
        assert_eq!(output_batch1.column(0).as_ref(), id_array1.as_ref());
        assert_eq!(
            output_batch1.column(1).as_ref(),
            expected_name_array1.as_ref()
        );

        // verify output of second batch
        let expected_name_array2 =
            Arc::new(StringArray::from(vec![Some("Bob"), Some("Bob")])) as ArrayRef;
        assert_eq!(output_batch2.column(0).as_ref(), id_array2.as_ref());
        assert_eq!(
            output_batch2.column(1).as_ref(),
            expected_name_array2.as_ref()
        );
    }
}
