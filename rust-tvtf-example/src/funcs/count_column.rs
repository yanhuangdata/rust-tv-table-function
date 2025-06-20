use std::{borrow::Cow, sync::Arc};

use anyhow::Context;
use arrow::array::{RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};

use crate::TableFunction;
use rust_tvtf_api::arg::{Arg, Args};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct CountColumn {
    pub output_column_name: Cow<'static, str>,
    columns_count: AtomicU64,
}

impl Default for CountColumn {
    fn default() -> Self {
        CountColumn {
            output_column_name: "column_count".into(),
            columns_count: AtomicU64::new(0),
        }
    }
}

impl CountColumn {
    pub fn new(params: Option<Args>) -> anyhow::Result<CountColumn> {
        let Some(params) = params else {
            return Ok(Self::default());
        };

        let scalars = params
            .into_iter()
            .filter(|p| p.is_scalar())
            .collect::<Vec<_>>();

        match scalars.len() {
            0 => Ok(Self::default()),
            1 => {
                let arg0 = scalars.first().unwrap();
                match arg0 {
                    Arg::String(output_column_name) => Ok(CountColumn {
                        output_column_name: output_column_name.to_string().into(),
                        columns_count: AtomicU64::new(0),
                    }),
                    _ => Err(anyhow::anyhow!("Parameter must be a string (column name).")),
                }
            }
            n => Err(anyhow::anyhow!(
                "Invalid arguments, there is no {n}-args constructor"
            )),
        }
    }
}

impl TableFunction for CountColumn {
    fn process(&mut self, input: RecordBatch) -> anyhow::Result<Option<RecordBatch>> {
        let num_columns = input.num_columns() as u64;
        self.columns_count.fetch_add(num_columns, Ordering::Relaxed);
        Ok(None)
    }

    fn finalize(&mut self) -> anyhow::Result<Option<RecordBatch>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("metric", DataType::Utf8, false),
            Field::new(self.output_column_name.clone(), DataType::UInt64, false),
        ]));
        let total_columns = self.columns_count.load(Ordering::Acquire);
        let metric_array = Arc::new(StringArray::from(vec!["total_columns"]));
        let count_array = Arc::new(UInt64Array::from(vec![total_columns]));
        let result_batch = RecordBatch::try_new(schema, vec![metric_array, count_array])
            .context("Failed to create result batch")?;
        Ok(Some(result_batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, Int32Array, StringArray};
    use arrow::record_batch::RecordBatch;
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_count_column_default() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("col1", DataType::Int32, false),
            Field::new("col2", DataType::Utf8, false),
        ]));
        let col1 = Arc::new(Int32Array::from(vec![1, 2])) as Arc<dyn Array>;
        let col2 = Arc::new(StringArray::from(vec!["a", "b"])) as Arc<dyn Array>;
        let input_batch =
            RecordBatch::try_new(schema, vec![col1, col2]).expect("Failed to create record batch");

        let mut count_column = CountColumn::default();

        // Process the batch
        let process_result = count_column
            .process(input_batch)
            .expect("Processing failed");
        assert!(process_result.is_none()); // Should always return None

        // Finalize to get the count
        let finalize_result = count_column.finalize().expect("Finalize failed");
        assert!(finalize_result.is_some());
        let result_batch = finalize_result.unwrap();

        assert_eq!(result_batch.num_columns(), 2);
        assert_eq!(result_batch.num_rows(), 1);

        let metric_array = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(metric_array.value(0), "total_columns");

        let count_array = result_batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(count_array.value(0), 2); // 2 columns counted

        assert_eq!(result_batch.schema().field(1).name(), "column_count");
    }

    #[test]
    fn test_count_column_multiple_batches() {
        let mut count_column = CountColumn::default();

        // First batch with 3 columns
        let schema1 = Arc::new(Schema::new(vec![
            Field::new("col1", DataType::Int32, false),
            Field::new("col2", DataType::Utf8, false),
            Field::new("col3", DataType::Float64, false),
        ]));
        let col1 = Arc::new(Int32Array::from(vec![1])) as Arc<dyn Array>;
        let col2 = Arc::new(StringArray::from(vec!["a"])) as Arc<dyn Array>;
        let col3 = Arc::new(Float64Array::from(vec![1.0])) as Arc<dyn Array>;
        let batch1 = RecordBatch::try_new(schema1, vec![col1, col2, col3])
            .expect("Failed to create record batch");

        // Second batch with 2 columns
        let schema2 = Arc::new(Schema::new(vec![
            Field::new("col_a", DataType::Int32, false),
            Field::new("col_b", DataType::Utf8, false),
        ]));
        let col_a = Arc::new(Int32Array::from(vec![1, 2])) as Arc<dyn Array>;
        let col_b = Arc::new(StringArray::from(vec!["x", "y"])) as Arc<dyn Array>;
        let batch2 = RecordBatch::try_new(schema2, vec![col_a, col_b])
            .expect("Failed to create record batch");

        // Process both batches
        let result1 = count_column
            .process(batch1)
            .expect("Processing batch1 failed");
        assert!(result1.is_none());

        let result2 = count_column
            .process(batch2)
            .expect("Processing batch2 failed");
        assert!(result2.is_none());

        // Finalize to get the total count
        let finalize_result = count_column.finalize().expect("Finalize failed");
        assert!(finalize_result.is_some());
        let result_batch = finalize_result.unwrap();

        let count_array = result_batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(count_array.value(0), 5); // 3 + 2 = 5 columns total
    }

    #[test]
    fn test_count_column_custom_name() {
        let schema = Arc::new(Schema::new(vec![Field::new("data", DataType::Utf8, false)]));
        let col1 = Arc::new(StringArray::from(vec!["test"])) as Arc<dyn Array>;
        let input_batch =
            RecordBatch::try_new(schema, vec![col1]).expect("Failed to create record batch");

        let params = vec![Arg::String("total_cols".to_string())];
        let mut count_column =
            CountColumn::new(Some(params)).expect("Failed to create CountColumn");

        let _result = count_column
            .process(input_batch)
            .expect("Processing failed");
        let finalize_result = count_column.finalize().expect("Finalize failed");
        let result_batch = finalize_result.unwrap();

        assert_eq!(result_batch.schema().field(1).name(), "total_cols");
        let count_array = result_batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(count_array.value(0), 1);
    }

    #[test]
    fn test_count_column_empty_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "col1",
            DataType::Int32,
            false,
        )]));
        let col1 = Arc::new(Int32Array::from(Vec::<i32>::new())) as Arc<dyn Array>;
        let input_batch =
            RecordBatch::try_new(schema, vec![col1]).expect("Failed to create record batch");

        let mut count_column = CountColumn::default();
        let _result = count_column
            .process(input_batch)
            .expect("Processing failed");
        let finalize_result = count_column.finalize().expect("Finalize failed");
        let result_batch = finalize_result.unwrap();

        let count_array = result_batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(count_array.value(0), 1); // Still counts 1 column even if empty
    }

    #[test]
    fn test_count_column_no_batches_processed() {
        let mut count_column = CountColumn::default();
        let finalize_result = count_column.finalize().expect("Finalize failed");
        let result_batch = finalize_result.unwrap();

        let count_array = result_batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(count_array.value(0), 0); // No batches processed = 0 columns
    }

    #[test]
    fn test_count_column_new_no_params() {
        let count_column =
            CountColumn::new(None).expect("Failed to create CountColumn with no params");
        assert_eq!(count_column.output_column_name, "column_count");
    }

    #[test]
    fn test_count_column_new_invalid_param_type() {
        let params = vec![Arg::Int(123)];
        let result = CountColumn::new(Some(params));
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Parameter must be a string (column name)."
        );
    }

    #[test]
    fn test_count_column_new_too_many_params() {
        let params = vec![
            Arg::String("count".to_string()),
            Arg::String("extra".to_string()),
        ];
        let result = CountColumn::new(Some(params));
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Invalid arguments, there is no 2-args constructor"
        );
    }
}
