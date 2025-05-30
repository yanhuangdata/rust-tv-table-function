use std::{borrow::Cow, sync::Arc};

use anyhow::Context;
use arrow::{
    array::{Array, Float64Array, Int64Array, RecordBatch, UInt64Array},
    compute::{cast, kernels::numeric::add},
};
use arrow_schema::{DataType, Field, Schema};

use crate::{
    arg::{Arg, Args},
    TableFunction,
};

#[derive(Debug)]
pub struct AddTotals {
    pub output_column_name: Cow<'static, str>,
}

impl Default for AddTotals {
    fn default() -> Self {
        AddTotals {
            output_column_name: "total".into(),
        }
    }
}

impl AddTotals {
    pub fn new(params: Option<Args>) -> anyhow::Result<AddTotals> {
        let Some(params) = params else {
            return Ok(Self::default());
        };
        let scalars = params
            .into_iter()
            .filter(|p| p.is_scalar())
            .collect::<Vec<_>>();
        if scalars.len() != 1 {
            return Ok(Self::default());
        }
        let arg0 = scalars.first()
            .context("No `output_column_name` parameter provided.")?;
        let Arg::String(output_column_name) = arg0 else {
            return Err(anyhow::anyhow!("`output_column_name` must be a string."));
        };

        Ok(AddTotals {
            output_column_name: output_column_name.to_string().into(),
        })
    }
}

impl TableFunction for AddTotals {
    fn process(&mut self, input: RecordBatch) -> anyhow::Result<Option<RecordBatch>> {
        use DataType as DT;
        if !input
            .columns()
            .iter()
            .any(|col| col.data_type().is_numeric())
        {
            return Ok(Some(input));
        }

        let use_signed = || {
            input.columns().iter().any(|col| {
                matches!(
                    col.data_type(),
                    DT::Int8 | DT::Int16 | DT::Int32 | DT::Int64
                )
            })
        };
        let use_float = || {
            input.columns().iter().any(|col| {
                matches!(
                    col.data_type(),
                    DT::Float16
                        | DT::Float32
                        | DT::Float64
                        | DT::Decimal128(_, _)
                        | DT::Decimal256(_, _)
                )
            })
        };

        let (mut total_array, data_type): (Arc<dyn Array>, DataType) = if use_float() {
            (
                Arc::new(Float64Array::from(vec![0.0; input.num_rows()])),
                DataType::Float64,
            )
        } else if use_signed() {
            (
                Arc::new(Int64Array::from(vec![0; input.num_rows()])),
                DataType::Int64,
            )
        } else {
            (
                Arc::new(UInt64Array::from(vec![0; input.num_rows()])),
                DataType::UInt64,
            )
        };

        for column in input.columns() {
            if !column.data_type().is_numeric() {
                continue;
            }
            let mut column = Arc::clone(column);
            if *column.data_type() != data_type {
                let casted = cast(&column, &data_type).context("Cast failed")?;
                column = casted;
            }
            total_array = add(&total_array, &column).context("Overflowed")?;
        }

        let total_field = Field::new(self.output_column_name.clone(), data_type, false);
        let mut new_fields = input.schema().fields().to_vec();
        new_fields.push(total_field.into());
        let new_schema = Arc::new(Schema::new(new_fields));
        let mut new_columns = input.columns().to_vec();
        new_columns.push(total_array);
        let output =
            RecordBatch::try_new(new_schema.clone(), new_columns).context("failed to new")?;
        Ok(Some(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array, StringArray, UInt32Array};
    use arrow::record_batch::RecordBatch;
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_add_totals_basic() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("col1", DataType::Int32, false),
            Field::new("col2", DataType::Int32, false),
        ]));
        let col1 = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn Array>;
        let col2 = Arc::new(Int32Array::from(vec![10, 20, 30])) as Arc<dyn Array>;
        let input_batch =
            RecordBatch::try_new(schema, vec![col1, col2]).expect("Failed to create record batch");

        let mut add_totals = AddTotals::default();
        let output_batch_option = add_totals.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        assert_eq!(output_batch.num_columns(), 3);
        assert_eq!(output_batch.column(2).data_type(), &DataType::Int64); // Sum of Int32 should default to Int64
        let total_array = output_batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(total_array.values(), &[11, 22, 33]);
        assert_eq!(output_batch.schema().field(2).name(), "total");
    }

    #[test]
    fn test_add_totals_float() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("col1", DataType::Float64, false),
            Field::new("col2", DataType::Float64, false),
        ]));
        let col1 = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3])) as Arc<dyn Array>;
        let col2 = Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])) as Arc<dyn Array>;
        let input_batch =
            RecordBatch::try_new(schema, vec![col1, col2]).expect("Failed to create record batch");

        let mut add_totals = AddTotals::default();
        let output_batch_option = add_totals.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        assert_eq!(output_batch.num_columns(), 3);
        assert_eq!(output_batch.column(2).data_type(), &DataType::Float64);
        let total_array = output_batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(total_array.values(), &[11.1, 22.2, 33.3]);
        assert_eq!(output_batch.schema().field(2).name(), "total");
    }

    #[test]
    fn test_add_totals_mixed_types() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("col1", DataType::Int32, false),
            Field::new("col2", DataType::Float64, false),
            Field::new("col3", DataType::UInt32, false),
        ]));
        let col1 = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn Array>;
        let col2 = Arc::new(Float64Array::from(vec![10.5, 20.5, 30.5])) as Arc<dyn Array>;
        let col3 = Arc::new(UInt32Array::from(vec![100, 200, 300])) as Arc<dyn Array>;
        let input_batch = RecordBatch::try_new(schema, vec![col1, col2, col3])
            .expect("Failed to create record batch");

        let mut add_totals = AddTotals::default();
        let output_batch_option = add_totals.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        assert_eq!(output_batch.num_columns(), 4);
        assert_eq!(output_batch.column(3).data_type(), &DataType::Float64); // Should cast to Float64 due to presence of Float64
        let total_array = output_batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(total_array.values(), &[111.5, 222.5, 333.5]);
    }

    #[test]
    fn test_add_totals_with_non_numeric_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("col1", DataType::Int32, false),
            Field::new("col_str", DataType::Utf8, false),
            Field::new("col2", DataType::Float64, false),
        ]));
        let col1 = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn Array>;
        let col_str = Arc::new(StringArray::from(vec!["a", "b", "c"])) as Arc<dyn Array>;
        let col2 = Arc::new(Float64Array::from(vec![10.5, 20.5, 30.5])) as Arc<dyn Array>;
        let input_batch = RecordBatch::try_new(schema, vec![col1, col_str, col2])
            .expect("Failed to create record batch");

        let mut add_totals = AddTotals::default();
        let output_batch_option = add_totals.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        assert_eq!(output_batch.num_columns(), 4);
        assert_eq!(output_batch.column(3).data_type(), &DataType::Float64);
        let total_array = output_batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(total_array.values(), &[11.5, 22.5, 33.5]);
    }

    #[test]
    fn test_add_totals_empty_input() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "col1",
            DataType::Int32,
            false,
        )]));
        let col1 = Arc::new(Int32Array::from(Vec::<i32>::new())) as Arc<dyn Array>;
        let input_batch =
            RecordBatch::try_new(schema, vec![col1]).expect("Failed to create record batch");

        let mut add_totals = AddTotals::default();
        let output_batch_option = add_totals.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        assert_eq!(output_batch.num_columns(), 2);
        assert_eq!(output_batch.num_rows(), 0);
        assert_eq!(output_batch.column(1).data_type(), &DataType::Int64);
    }

    #[test]
    fn test_add_totals_no_numeric_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("col_str", DataType::Utf8, false),
            Field::new("col_bool", DataType::Boolean, false),
        ]));
        let col_str = Arc::new(StringArray::from(vec!["a", "b"])) as Arc<dyn Array>;
        let col_bool =
            Arc::new(arrow::array::BooleanArray::from(vec![true, false])) as Arc<dyn Array>;
        let input_batch = RecordBatch::try_new(schema, vec![col_str, col_bool])
            .expect("Failed to create record batch");

        let mut add_totals = AddTotals::default();
        let output_batch_option = add_totals
            .process(input_batch.clone()) // Use clone to compare later
            .expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        // If no numeric columns, the input batch should be returned as is.
        assert_eq!(output_batch.num_columns(), input_batch.num_columns());
        assert_eq!(output_batch.schema(), input_batch.schema());
        for i in 0..output_batch.num_columns() {
            assert_eq!(output_batch.column(i), input_batch.column(i));
        }
    }

    #[test]
    fn test_add_totals_custom_column_name() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "col1",
            DataType::Int32,
            false,
        )]));
        let col1 = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn Array>;
        let input_batch =
            RecordBatch::try_new(schema, vec![col1]).expect("Failed to create record batch");

        let params = vec![Arg::String("my_sum_column".to_string())];
        let mut add_totals = AddTotals::new(Some(params)).expect("Failed to create AddTotals");
        let output_batch_option = add_totals.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        assert_eq!(output_batch.num_columns(), 2);
        assert_eq!(output_batch.schema().field(1).name(), "my_sum_column");
    }

    #[test]
    fn test_add_totals_new_with_no_params() {
        let add_totals = AddTotals::new(None).expect("Failed to create AddTotals with no params");
        assert_eq!(add_totals.output_column_name, "total");
    }

    #[test]
    fn test_add_totals_new_with_incorrect_param_type() {
        let params = vec![Arg::Int(123)];
        let result = AddTotals::new(Some(params));
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "`output_column_name` must be a string."
        );
    }
}
