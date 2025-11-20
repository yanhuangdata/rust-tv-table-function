use std::sync::Arc;

use anyhow::Context;
use arrow::{
    array::{Array, ArrayRef, AsArray, RecordBatch},
    compute::take,
};
use arrow_schema::{DataType, Field, Schema};

use crate::TableFunction;
use rust_tvtf_api::arg::{Arg, Args};

#[derive(Debug)]
pub struct Flatten {
    column_names: Vec<String>,
}

impl Flatten {
    pub fn new(params: Option<Args>) -> anyhow::Result<Flatten> {
        let Some(params) = params else {
            return Err(anyhow::anyhow!("Column names parameter is required"));
        };

        let scalars = params
            .into_iter()
            .filter(|p| p.is_scalar())
            .collect::<Vec<_>>();

        if scalars.is_empty() {
            return Err(anyhow::anyhow!("At least one column name must be provided"));
        }

        let arg0 = scalars
            .first()
            .context("No column names parameter provided")?;

        let Arg::String(column_names_str) = arg0 else {
            return Err(anyhow::anyhow!("Column names must be a string"));
        };

        // Parse comma-separated column names
        let column_names: Vec<String> = column_names_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if column_names.is_empty() {
            return Err(anyhow::anyhow!("At least one column name must be provided"));
        }

        Ok(Flatten { column_names })
    }
}

impl TableFunction for Flatten {
    fn process(&mut self, input: RecordBatch) -> anyhow::Result<Option<RecordBatch>> {
        // Find the columns to flatten
        let schema = input.schema();
        let mut list_columns_info: Vec<(usize, Arc<dyn Array>)> = Vec::new();
        let mut processed_columns: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        let mut missing_column_encountered = false;

        for column_name in &self.column_names {
            let Some((column_index, field)) = schema.column_with_name(column_name) else {
                missing_column_encountered = true;
                continue;
            };

            if !matches!(field.data_type(), DataType::List(_) | DataType::Null) {
                return Err(anyhow::anyhow!(
                    "Column '{}' is not a list type or null type, got {:?}",
                    column_name,
                    field.data_type()
                ));
            }

            if !processed_columns.contains(&column_index) {
                list_columns_info.push((column_index, Arc::clone(input.column(column_index))));
                processed_columns.insert(column_index);
            }
        }

        if missing_column_encountered {
            let mut new_fields = schema.fields().to_vec();
            let mut updated_columns: std::collections::HashSet<usize> =
                std::collections::HashSet::new();

            for (col_idx, _) in &list_columns_info {
                if !updated_columns.contains(col_idx) {
                    let original_field = &new_fields[*col_idx];
                    let inner_type = match original_field.data_type() {
                        DataType::List(f) | DataType::LargeList(f) => f.data_type().clone(),
                        DataType::Null => DataType::Null,
                        _ => unreachable!(),
                    };
                    new_fields[*col_idx] = Arc::new(Field::new(
                        original_field.name(),
                        inner_type,
                        original_field.is_nullable(),
                    ));
                    updated_columns.insert(*col_idx);
                }
            }

            let new_schema = Arc::new(Schema::new(new_fields));
            return Ok(Some(RecordBatch::new_empty(new_schema)));
        }

        if list_columns_info.is_empty() {
            return Ok(Some(input));
        }

        let mut list_lengths: Vec<Vec<usize>> = Vec::new();

        for (_col_idx, column) in &list_columns_info {
            let lengths = match column.data_type() {
                DataType::Null => {
                    // For null type columns, treat each row as an empty list (length 0)
                    vec![0; column.len()]
                }
                DataType::List(_) => {
                    let list_array = column
                        .as_list_opt::<i32>()
                        .ok_or_else(|| anyhow::anyhow!("Failed to cast column to a ListArray"))?;
                    (0..list_array.len())
                        .map(|i| {
                            if list_array.is_valid(i) {
                                list_array.value(i).len()
                            } else {
                                1
                            }
                        }) // Treat null list as a single null element
                        .collect()
                }
                dt => {
                    return Err(anyhow::anyhow!(
                        "Unexpected data type in list_columns_info: {dt}"
                    ));
                }
            };
            list_lengths.push(lengths);
        }

        // Calculate the product of list lengths for each row (size of Cartesian product per row)
        let row_products: Vec<usize> = (0..input.num_rows())
            .map(|row_idx| {
                list_lengths
                    .iter()
                    .map(|lens| lens[row_idx])
                    .product()
            })
            .collect();

        let total_rows: usize = row_products.iter().sum();

        // Update the schema for the flattened columns
        let mut new_fields = schema.fields().to_vec();
        let mut updated_columns: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        for (col_idx, _) in &list_columns_info {
            if !updated_columns.contains(col_idx) {
                let original_field = &new_fields[*col_idx];
                let inner_type = match original_field.data_type() {
                    DataType::List(f) | DataType::LargeList(f) => f.data_type().clone(),
                    DataType::Null => DataType::Null, // For null type, keep it as null
                    _ => unreachable!(),
                };
                new_fields[*col_idx] = Arc::new(Field::new(
                    original_field.name(),
                    inner_type,
                    original_field.is_nullable(),
                ));
                updated_columns.insert(*col_idx);
            }
        }

        let new_schema = Arc::new(Schema::new(new_fields));

        if total_rows == 0 {
            return Ok(Some(RecordBatch::new_empty(new_schema)));
        }

        let mut new_columns: Vec<ArrayRef> = Vec::new();
        let mut processed_columns: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        for (col_idx, field) in schema.fields().iter().enumerate() {
            let column = input.column(col_idx);

            if let Some(list_col_pos) = list_columns_info
                .iter()
                .position(|(idx, _)| *idx == col_idx)
            {
                if !processed_columns.contains(&col_idx) {
                    let inner_type = match field.data_type() {
                        DataType::List(inner_field) | DataType::LargeList(inner_field) => {
                            inner_field.data_type()
                        }
                        DataType::Null => {
                            // For null type, we need to determine the appropriate inner type
                            // Since null arrays can be cast to any type, we'll use a generic approach
                            // that treats null as a valid inner type
                            &DataType::Null
                        }
                        _ => unreachable!(),
                    };

                    let mut flattened_values = Vec::new();

                    // Check if it's a null type field
                    if matches!(field.data_type(), DataType::Null) {
                        // For null type, we don't have actual values to flatten
                        // Just add nulls based on the cartesian product counts
                        for item in &row_products {
                            if *item == 0 {
                                continue;
                            }

                            for _ in 0..*item {
                                flattened_values.push(None);
                            }
                        }
                    } else {
                        // For actual list types, use the existing logic
                        let list_array = column.as_list_opt::<i32>().unwrap();

                        for row_idx in 0..input.num_rows() {
                            let row_product = row_products[row_idx];
                            if row_product == 0 {
                                continue;
                            }

                            if list_array.is_null(row_idx) {
                                for _ in 0..row_product {
                                    flattened_values.push(None);
                                }
                                continue;
                            }

                            let outer_repeat = list_lengths[..list_col_pos]
                                .iter()
                                .map(|lens| lens[row_idx])
                                .product::<usize>();

                            let inner_repeat = list_lengths[list_col_pos + 1..]
                                .iter()
                                .map(|lens| lens[row_idx])
                                .product::<usize>();

                            let repeats = outer_repeat * inner_repeat;
                            let list_values = list_array.value(row_idx);

                            if list_values.is_empty() {
                                for _ in 0..repeats {
                                    flattened_values.push(None);
                                }
                                continue;
                            }

                            // Apply the Cartesian product pattern
                            for _ in 0..outer_repeat {
                                for elem_idx in 0..list_values.len() {
                                    for _ in 0..inner_repeat {
                                        if list_values.is_null(elem_idx) {
                                            flattened_values.push(None);
                                        } else {
                                            // This is a generic way to append a value from one array to a builder
                                            flattened_values
                                                .push(Some(list_values.slice(elem_idx, 1)));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let flattened_array = build_array_from_slices(flattened_values, inner_type)?;
                    new_columns.push(flattened_array);
                    processed_columns.insert(col_idx);
                }
            } else {
                let mut indices = Vec::with_capacity(total_rows);
                for (row_idx, &product) in row_products.iter().enumerate() {
                    indices.resize(indices.len() + product, row_idx as u32);
                }
                let indices_array = arrow::array::UInt32Array::from(indices);
                let repeated_column = take(column.as_ref(), &indices_array, None)?;
                new_columns.push(repeated_column);
            }
        }

        // Update the schema for the flattened columns
        let mut new_fields = schema.fields().to_vec();
        let mut updated_columns: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        for (col_idx, _) in &list_columns_info {
            if !updated_columns.contains(col_idx) {
                let original_field = &new_fields[*col_idx];
                let inner_type = match original_field.data_type() {
                    DataType::List(f) | DataType::LargeList(f) => f.data_type().clone(),
                    DataType::Null => DataType::Null, // For null type, keep it as null
                    _ => unreachable!(),
                };
                new_fields[*col_idx] = Arc::new(Field::new(
                    original_field.name(),
                    inner_type,
                    original_field.is_nullable(),
                ));
                updated_columns.insert(*col_idx);
            }
        }

        let new_schema = Arc::new(Schema::new(new_fields));
        let expected_rows = total_rows;
        let column_lengths: Vec<usize> = new_columns.iter().map(|col| col.len()).collect();
        let schema_field_count = new_schema.fields().len();
        let column_count = new_columns.len();

        let output = RecordBatch::try_new(Arc::clone(&new_schema), new_columns).with_context(
            || {
                format!(
                    "Failed to create output RecordBatch (fields: {}, columns: {}, expected rows: {}, column lengths: {:?})",
                    schema_field_count, column_count, expected_rows, column_lengths
                )
            },
        )?;

        Ok(Some(output))
    }
}

fn build_array_from_slices(
    slices: Vec<Option<ArrayRef>>,
    data_type: &DataType,
) -> anyhow::Result<ArrayRef> {
    use arrow::array::*;

    // Concatenate all non-null slices and track null positions
    let mut arrays_to_concat = Vec::new();
    let mut null_bitmap = Vec::new();

    for slice_opt in slices {
        if let Some(slice) = slice_opt {
            arrays_to_concat.push(slice);
            null_bitmap.push(true);
        } else {
            // Create a null array of the appropriate type
            let null_array = new_null_array(data_type, 1);
            arrays_to_concat.push(null_array);
            null_bitmap.push(false);
        }
    }

    if arrays_to_concat.is_empty() {
        return Ok(new_null_array(data_type, 0));
    }

    // Concatenate arrays
    let refs: Vec<&dyn Array> = arrays_to_concat.iter().map(|a| a.as_ref()).collect();
    arrow::compute::concat(&refs).context("Failed to concatenate arrays")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, ListArray, StringArray};
    use arrow::buffer::OffsetBuffer;
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_flatten_basic() {
        // Create a list column: [[1,2], [3,4,5]]
        let values = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        let offsets = OffsetBuffer::from_lengths([2, 3]);
        let list_array = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            offsets,
            values,
            None,
        ));

        // Create another regular column
        let string_array = Arc::new(StringArray::from(vec!["a", "b"]));

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "list_col",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
            Field::new("str_col", DataType::Utf8, false),
        ]));

        let input_batch = RecordBatch::try_new(
            schema,
            vec![list_array as Arc<dyn Array>, string_array as Arc<dyn Array>],
        )
        .expect("Failed to create record batch");

        let params = vec![Arg::String("list_col".to_string())];
        let mut flatten = Flatten::new(Some(params)).expect("Failed to create Flatten");
        let output_batch_option = flatten.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        // Should have 5 rows (2 + 3)
        assert_eq!(output_batch.num_rows(), 5);
        assert_eq!(output_batch.num_columns(), 2);

        // Check flattened values
        let flattened_col = output_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(flattened_col.values(), &[1, 2, 3, 4, 5]);

        // Check repeated string values
        let str_col = output_batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(str_col.value(0), "a");
        assert_eq!(str_col.value(1), "a");
        assert_eq!(str_col.value(2), "b");
        assert_eq!(str_col.value(3), "b");
        assert_eq!(str_col.value(4), "b");
    }

    #[test]
    fn test_flatten_multiple_columns() {
        // Create first list column: [[1,2], [3]]
        let values1 = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let offsets1 = OffsetBuffer::from_lengths([2, 1]);
        let list_array1 = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            offsets1,
            values1,
            None,
        ));

        // Create second list column: [[10,20], [30,40]]
        let values2 = Arc::new(Int32Array::from(vec![10, 20, 30, 40]));
        let offsets2 = OffsetBuffer::from_lengths([2, 2]);
        let list_array2 = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            offsets2,
            values2,
            None,
        ));

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "list1",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
            Field::new(
                "list2",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
        ]));

        let input_batch = RecordBatch::try_new(
            schema,
            vec![list_array1 as Arc<dyn Array>, list_array2 as Arc<dyn Array>],
        )
        .expect("Failed to create record batch");

        let params = vec![Arg::String("list1,list2".to_string())];
        let mut flatten = Flatten::new(Some(params)).expect("Failed to create Flatten");
        let output_batch_option = flatten.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        dbg!(&output_batch);

        // Should have 6 rows (2*2 + 1*2)
        assert_eq!(output_batch.num_rows(), 6);

        // Check Cartesian product results
        let col1 = output_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let col2 = output_batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        // First row combinations: [1,2] x [10,20]
        assert_eq!(col1.value(0), 1);
        assert_eq!(col2.value(0), 10);
        assert_eq!(col1.value(1), 1);
        assert_eq!(col2.value(1), 20);
        assert_eq!(col1.value(2), 2);
        assert_eq!(col2.value(2), 10);
        assert_eq!(col1.value(3), 2);
        assert_eq!(col2.value(3), 20);

        // Second row combinations: [3] x [30,40]
        assert_eq!(col1.value(4), 3);
        assert_eq!(col2.value(4), 30);
        assert_eq!(col1.value(5), 3);
        assert_eq!(col2.value(5), 40);
    }

    #[test]
    fn test_flatten_empty_list() {
        // Create a list column with an empty list: [[], [1,2]]
        let values = Arc::new(Int32Array::from(vec![1, 2]));
        let offsets = OffsetBuffer::from_lengths([0, 2]);
        let list_array = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            offsets,
            values,
            None,
        ));

        let schema = Arc::new(Schema::new(vec![Field::new(
            "list_col",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]));

        let input_batch = RecordBatch::try_new(schema, vec![list_array as Arc<dyn Array>])
            .expect("Failed to create record batch");

        let params = vec![Arg::String("list_col".to_string())];
        let mut flatten = Flatten::new(Some(params)).expect("Failed to create Flatten");
        let output_batch_option = flatten.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        // Should have 2 rows (0 + 2)
        assert_eq!(output_batch.num_rows(), 2);

        let flattened_col = output_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(flattened_col.values(), &[1, 2]);
    }

    #[test]
    fn test_flatten_no_columns() {
        let result = Flatten::new(None);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Column names parameter is required"
        );
    }

    #[test]
    fn test_flatten_invalid_column_type() {
        let params = vec![Arg::Int(123)];
        let result = Flatten::new(Some(params));
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Column names must be a string"
        );
    }

    #[test]
    fn test_flatten_non_list_column() {
        let int_array = Arc::new(Int32Array::from(vec![1, 2, 3]));

        let schema = Arc::new(Schema::new(vec![Field::new(
            "int_col",
            DataType::Int32,
            false,
        )]));

        let input_batch = RecordBatch::try_new(schema, vec![int_array as Arc<dyn Array>])
            .expect("Failed to create record batch");

        let params = vec![Arg::String("int_col".to_string())];
        let mut flatten = Flatten::new(Some(params)).expect("Failed to create Flatten");
        let result = flatten.process(input_batch);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("is not a list type")
        );
    }

    #[test]
    fn test_flatten_column_not_found() {
        let int_array = Arc::new(Int32Array::from(vec![1, 2, 3]));

        let schema = Arc::new(Schema::new(vec![Field::new(
            "int_col",
            DataType::Int32,
            false,
        )]));
        let original_schema = schema.clone();

        let input_batch = RecordBatch::try_new(schema, vec![int_array as Arc<dyn Array>])
            .expect("Failed to create record batch");

        let params = vec![Arg::String("nonexistent_col".to_string())];
        let mut flatten = Flatten::new(Some(params)).expect("Failed to create Flatten");
        let output_batch_option = flatten.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        assert_eq!(output_batch.num_rows(), 0);
        assert_eq!(output_batch.num_columns(), original_schema.fields().len());

        let output_schema = output_batch.schema();
        assert_eq!(output_schema.fields().len(), original_schema.fields().len());
        assert_eq!(output_schema.field(0).name(), "int_col");
        assert_eq!(output_schema.field(0).data_type(), &DataType::Int32);
    }

    #[test]
    fn test_flatten_duplicate_columns() {
        // Create a list column: [[1,2], [3,4]]
        let values = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let offsets = OffsetBuffer::from_lengths([2, 2]);
        let list_array = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            offsets,
            values,
            None,
        ));

        let schema = Arc::new(Schema::new(vec![Field::new(
            "list_col",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]));

        let input_batch = RecordBatch::try_new(schema, vec![list_array as Arc<dyn Array>])
            .expect("Failed to create record batch");

        // Test with duplicate column names
        let params = vec![Arg::String("list_col,list_col".to_string())];
        let mut flatten = Flatten::new(Some(params)).expect("Failed to create Flatten");
        let output_batch_option = flatten.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        // Should have 4 rows (2 + 2)
        assert_eq!(output_batch.num_rows(), 4);
        assert_eq!(output_batch.num_columns(), 1);

        // Check flattened values
        let flattened_col = output_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(flattened_col.values(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_flatten_empty_list_in_multi_column_row_skips_row() {
        // list1: [[Alice, Bob], [Charlie, David]]
        let user_values =
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie", "David"])) as ArrayRef;
        let user_offsets = OffsetBuffer::from_lengths([2, 2]);
        let user_list = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Utf8, true)),
            user_offsets,
            user_values,
            None,
        ));

        // list2: [[], [Game, Music]]
        let tag_values = Arc::new(StringArray::from(vec!["Game", "Music"])) as ArrayRef;
        let tag_offsets = OffsetBuffer::from_lengths([0, 2]);
        let tag_list = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Utf8, true)),
            tag_offsets,
            tag_values,
            None,
        ));

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "user",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                "tag",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
        ]));

        let input_batch = RecordBatch::try_new(
            schema,
            vec![user_list as Arc<dyn Array>, tag_list as Arc<dyn Array>],
        )
        .expect("Failed to create record batch");

        let params = vec![Arg::String("user,tag".to_string())];
        let mut flatten = Flatten::new(Some(params)).expect("Failed to create Flatten");
        let output_batch_option = flatten.process(input_batch).expect("Processing failed");
        let output_batch = output_batch_option.expect("Expected non-empty RecordBatch");

        // First row had an empty tag list, so only the second row remains (2 users x 2 tags)
        assert_eq!(output_batch.num_rows(), 4);

        let user_col = output_batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(user_col.value(0), "Charlie");
        assert_eq!(user_col.value(1), "Charlie");
        assert_eq!(user_col.value(2), "David");
        assert_eq!(user_col.value(3), "David");

        let tag_col = output_batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(tag_col.value(0), "Game");
        assert_eq!(tag_col.value(1), "Music");
        assert_eq!(tag_col.value(2), "Game");
        assert_eq!(tag_col.value(3), "Music");
    }

    #[test]
    fn test_flatten_null_type_as_empty() {
        use arrow::array::NullArray;
        // Create a null type column with 2 rows (should be treated as empty lists)
        let null_array = Arc::new(NullArray::new(2));

        let schema = Arc::new(Schema::new(vec![Field::new(
            "null_col",
            DataType::Null,
            true,
        )]));

        let input_batch = RecordBatch::try_new(schema, vec![null_array as Arc<dyn Array>])
            .expect("Failed to create record batch");

        let params = vec![Arg::String("null_col".to_string())];
        let mut flatten = Flatten::new(Some(params)).expect("Failed to create Flatten");
        let output_batch_option = flatten.process(input_batch).expect("Processing failed");
        assert!(output_batch_option.is_some());
        let output_batch = output_batch_option.unwrap();

        // Should have 0 rows because null type is treated as empty lists
        assert_eq!(output_batch.num_rows(), 0);
        assert_eq!(output_batch.num_columns(), 1);

        // Check that the output column also has null type
        assert_eq!(output_batch.schema().field(0).data_type(), &DataType::Null);
    }
}
