use crate::{PathTrav, TableFunction, get_external_dir};
use anyhow::{Context, anyhow};
use arrow::{
    array::{StringArray, UInt64Array},
    csv::Writer,
    record_batch::RecordBatch,
};
use arrow_csv::WriterBuilder;
use arrow_schema::{DataType, Field, Schema};
use parking_lot::Mutex;
use rust_tvtf_api::arg::{Arg, Args};
use std::{
    fs::File,
    io::BufWriter,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Debug)]
pub struct OutputCsv {
    pub target_path: PathBuf,
    pub tee: bool,
    pub append: bool,
    writer: Mutex<Writer<BufWriter<File>>>,
    rows_written: AtomicU64,
}

impl OutputCsv {
    pub fn new(args: Option<Args>, named_args: Vec<(String, Arg)>) -> anyhow::Result<OutputCsv> {
        let Some(params) = args else {
            return Err(anyhow::anyhow!(
                "`output_csv` requires at least a `target_path` parameter."
            ));
        };

        let scalars = params
            .into_iter()
            .filter(|p| p.is_scalar())
            .collect::<Vec<_>>();
        // arg 0
        let target_path = || {
            let target_path = scalars
                .first()
                .context("No `target_path` parameter provided.")?;
            let Arg::String(target_path) = target_path else {
                return Err(anyhow::anyhow!("`target_path` must be a string."));
            };
            Ok(target_path)
        };
        let mut append = false;
        let mut tee = false;
        for (name, arg) in named_args {
            match name.as_str() {
                "append" => {
                    let val = match arg {
                        Arg::Bool(i) => i,
                        _ => {
                            return Err(anyhow!("Invalid type for {}. Expected boolean.", name));
                        }
                    };
                    append = val;
                }
                "tee" => {
                    let val = match arg {
                        Arg::Bool(i) => i,
                        _ => {
                            return Err(anyhow!("Invalid type for {}. Expected boolean.", name));
                        }
                    };
                    tee = val;
                }
                _ => {
                    return Err(anyhow!(
                        "Invalid named parameter for trans table function: {}",
                        name
                    ));
                }
            }
        }

        if scalars.len() != 1 {
            return Err(anyhow!(
                "Invalid arguments, there is no {}-args constructor",
                scalars.len()
            ));
        }

        let target_path = target_path()?;
        let mut target_path = PathBuf::from(target_path);

        if let Some(external_dir) = get_external_dir() {
            if target_path.is_relative() {
                target_path = external_dir.join(target_path);
            }

            if let Ok(false) = target_path.is_path_trav(&external_dir) {
                return Err(anyhow!(
                    "{} does not exist in {}",
                    target_path.to_string_lossy(),
                    external_dir.to_string_lossy()
                ));
            }
        }

        // Handle append vs create logic
        let file = if append {
            File::options()
                .create(true)
                .append(true)
                .open(&target_path)
                .context(format!(
                    "Failed to create/open file in append mode at: {}",
                    target_path.to_string_lossy()
                ))?
        } else {
            File::create(&target_path).context(format!(
                "Failed to create/open file at: {}",
                target_path.to_string_lossy()
            ))?
        };

        let should_write_headers = if append {
            file.metadata().map(|meta| meta.len() == 0).unwrap_or(true) // Default to writing headers if we can't determine file size
        } else {
            true
        };

        let writer = WriterBuilder::new()
            .with_header(should_write_headers)
            .build(BufWriter::new(file));

        Ok(OutputCsv {
            target_path: target_path.to_path_buf(),
            writer: writer.into(),
            tee,
            append,
            rows_written: 0.into(),
        })
    }
}

impl TableFunction for OutputCsv {
    fn process(&mut self, input: RecordBatch) -> anyhow::Result<Option<RecordBatch>> {
        let mut writer = self.writer.lock();
        writer.write(&input)?;
        self.rows_written.fetch_add(
            input.num_rows() as u64,
            // `Relaxed` here because we don't need ordering constraints during computation.
            // We only use this variable for metrics.
            Ordering::Relaxed,
        );
        if self.tee {
            return Ok(Some(input));
        }
        Ok(None)
    }

    fn finalize(&mut self) -> anyhow::Result<Option<RecordBatch>> {
        if self.tee {
            return Ok(None);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("target_path", DataType::Utf8, false),
            Field::new("status", DataType::Utf8, false),
            Field::new("rows_written", DataType::UInt64, false),
        ]));

        let target_path_array = Arc::new(StringArray::from(vec![
            self.target_path.to_string_lossy().into_owned(),
        ]));
        let status_array = Arc::new(StringArray::from(vec!["completed"]));
        let rows_written = self.rows_written.load(Ordering::Acquire);
        let rows_written_array = Arc::new(UInt64Array::from(vec![rows_written]));

        let status_batch = RecordBatch::try_new(
            schema,
            vec![target_path_array, status_array, rows_written_array],
        )
        .context("Failed to create status record batch")?;

        Ok(Some(status_batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Int32Array, StringArray},
        record_batch::RecordBatch,
    };
    use arrow_schema::{DataType, Field, Schema};
    use std::{fs, io::Write, sync::Arc};
    use tempfile::NamedTempFile;

    // Helper function to create a simple RecordBatch
    fn create_sample_record_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<Option<i32>> = (0..rows).map(|i| Some(i as i32)).collect();
        let names: Vec<Option<String>> = (0..rows).map(|i| Some(format!("Name_{}", i))).collect();

        let id_array = Arc::new(Int32Array::from(ids)) as Arc<dyn arrow::array::Array>;
        let name_array = Arc::new(StringArray::from(names)) as Arc<dyn arrow::array::Array>;

        RecordBatch::try_new(schema, vec![id_array, name_array])
            .expect("Failed to create sample record batch")
    }

    #[test]
    fn test_output_csv_successful_write() {
        let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        let path = temp_file.path().to_str().unwrap().to_string();

        let mut output_csv = OutputCsv::new(Some(vec![Arg::String(path.clone())]), vec![])
            .expect("Failed to create OutputCsv instance");

        let batch1 = create_sample_record_batch(2);
        let batch2 = create_sample_record_batch(3);

        let result1 = output_csv
            .process(batch1)
            .expect("Processing batch 1 failed");
        assert!(result1.is_none(), "Process should return None");

        let result2 = output_csv
            .process(batch2)
            .expect("Processing batch 2 failed");
        assert!(result2.is_none(), "Process should return None");

        let finalize_result = output_csv.finalize().expect("Finalize failed");
        assert!(
            finalize_result.is_some(),
            "Finalize should return a status batch"
        );
        let status_batch = finalize_result.unwrap();

        assert_eq!(status_batch.num_columns(), 3);
        assert_eq!(status_batch.num_rows(), 1);

        let status_path = status_batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(status_path, path);

        let status_msg = status_batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(status_msg, "completed");

        let rows_written = status_batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);
        assert_eq!(rows_written, 5); // 2 from batch1 + 3 from batch2

        let file_content = fs::read_to_string(&path).expect("Failed to read CSV file");
        let expected_content = "id,name\n0,Name_0\n1,Name_1\n0,Name_0\n1,Name_1\n2,Name_2\n";
        assert_eq!(file_content, expected_content);
    }

    #[test]
    fn test_output_csv_append_mode() {
        let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        let path = temp_file.path().to_str().unwrap().to_string();

        // First, write some initial data
        {
            let mut output_csv = OutputCsv::new(Some(vec![Arg::String(path.clone())]), vec![])
                .expect("Failed to create OutputCsv instance");

            let batch = create_sample_record_batch(2);
            output_csv.process(batch).expect("Processing failed");
            output_csv.finalize().expect("Finalize failed");
        }

        // Now append more data
        {
            let mut output_csv = OutputCsv::new(
                Some(vec![Arg::String(path.clone())]),
                vec![("append".to_string(), Arg::Bool(true))],
            )
            .expect("Failed to create OutputCsv instance in append mode");

            let batch = create_sample_record_batch(2);
            output_csv.process(batch).expect("Processing failed");
            let finalize_result = output_csv.finalize().expect("Finalize failed");

            let status_batch = finalize_result.unwrap();
            let rows_written = status_batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0);
            assert_eq!(rows_written, 2); // Only the rows written in append mode
        }

        let file_content = fs::read_to_string(&path).expect("Failed to read CSV file");
        // Should have headers once, then original data, then appended data
        let expected_content = "id,name\n0,Name_0\n1,Name_1\n0,Name_0\n1,Name_1\n";
        assert_eq!(file_content, expected_content);
    }

    #[test]
    fn test_output_csv_append_to_empty_file() {
        let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        let path = temp_file.path().to_str().unwrap().to_string();

        let mut output_csv = OutputCsv::new(
            Some(vec![Arg::String(path.clone())]),
            vec![("append".to_string(), Arg::Bool(true))],
        )
        .expect("Failed to create OutputCsv instance in append mode");

        let batch = create_sample_record_batch(2);
        output_csv.process(batch).expect("Processing failed");
        output_csv.finalize().expect("Finalize failed");

        let file_content = fs::read_to_string(&path).expect("Failed to read CSV file");
        // Should include headers since file was empty
        let expected_content = "id,name\n0,Name_0\n1,Name_1\n";
        assert_eq!(file_content, expected_content);
    }

    #[test]
    fn test_output_csv_append_to_existing_file_with_content() {
        let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        let path = temp_file.path().to_str().unwrap().to_string();

        // Pre-populate the file with some CSV content
        {
            let mut file = File::create(&path).expect("Failed to create file");
            writeln!(file, "id,name").expect("Failed to write header");
            writeln!(file, "100,Existing_Name").expect("Failed to write data");
        }

        let mut output_csv = OutputCsv::new(
            Some(vec![Arg::String(path.clone())]),
            vec![("append".to_string(), Arg::Bool(true))],
        )
        .expect("Failed to create OutputCsv instance in append mode");

        let batch = create_sample_record_batch(2);
        output_csv.process(batch).expect("Processing failed");
        output_csv.finalize().expect("Finalize failed");

        let file_content = fs::read_to_string(&path).expect("Failed to read CSV file");
        // Should not duplicate headers, just append data
        let expected_content = "id,name\n100,Existing_Name\n0,Name_0\n1,Name_1\n";
        assert_eq!(file_content, expected_content);
    }

    #[test]
    fn test_output_csv_new_invalid_path() {
        let invalid_path = "/nonexistent_dir/invalid/file.csv";
        let result = OutputCsv::new(Some(vec![Arg::String(invalid_path.to_string())]), vec![]);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to create/open file at:"),
            "Error message should indicate file creation failure"
        );
        assert!(
            err_msg.contains(invalid_path),
            "Error message should contain the invalid path"
        );
    }

    #[test]
    fn test_output_csv_new_no_params() {
        let result = OutputCsv::new(None, vec![]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "`output_csv` requires at least a `target_path` parameter."
        );
    }

    #[test]
    fn test_output_csv_new_incorrect_param_type() {
        let params = vec![Arg::Int(123)];
        let result = OutputCsv::new(Some(params), vec![]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "`target_path` must be a string."
        );
    }

    #[test]
    fn test_output_csv_no_numeric_columns_in_input() {
        let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        let path = temp_file.path().to_str().unwrap().to_string();

        let mut output_csv = OutputCsv::new(Some(vec![Arg::String(path.clone())]), vec![])
            .expect("Failed to create OutputCsv instance");

        let schema = Arc::new(Schema::new(vec![
            Field::new("col_str", DataType::Utf8, false),
            Field::new("col_bool", DataType::Boolean, false),
        ]));
        let col_str =
            Arc::new(StringArray::from(vec!["apple", "banana"])) as Arc<dyn arrow::array::Array>;
        let col_bool = Arc::new(arrow::array::BooleanArray::from(vec![true, false]))
            as Arc<dyn arrow::array::Array>;
        let input_batch = RecordBatch::try_new(schema, vec![col_str, col_bool])
            .expect("Failed to create record batch");

        let result = output_csv.process(input_batch).expect("Processing failed");
        assert!(result.is_none());

        let finalize_result = output_csv.finalize().expect("Finalize failed");
        assert!(finalize_result.is_some());
        let status_batch = finalize_result.unwrap();
        let rows_written = status_batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);
        assert_eq!(rows_written, 2);

        let file_content = fs::read_to_string(&path).expect("Failed to read CSV file");
        let expected_content = "col_str,col_bool\napple,true\nbanana,false\n";
        assert_eq!(file_content, expected_content);
    }

    #[test]
    fn test_output_csv_different_schema_in_input() {
        let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        let path = temp_file.path().to_str().unwrap().to_string();

        let mut output_csv = OutputCsv::new(Some(vec![Arg::String(path.clone())]), vec![])
            .expect("Failed to create OutputCsv instance");

        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("col_str", DataType::Utf8, false),
                Field::new("col_bool", DataType::Boolean, false),
            ]));
            let col_str = Arc::new(StringArray::from(vec!["apple", "banana"]))
                as Arc<dyn arrow::array::Array>;
            let col_bool = Arc::new(arrow::array::BooleanArray::from(vec![true, false]))
                as Arc<dyn arrow::array::Array>;
            let input_batch = RecordBatch::try_new(schema, vec![col_str, col_bool])
                .expect("Failed to create record batch");

            let result = output_csv.process(input_batch);
            assert!(result.is_ok())
        }
        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("col_int", DataType::Int32, false),
                Field::new("col_bool", DataType::Boolean, false),
                Field::new("col_str", DataType::Utf8, false),
            ]));
            let col_int = Arc::new(Int32Array::from(vec![1, 2])) as Arc<dyn arrow::array::Array>;
            let col_bool = Arc::new(arrow::array::BooleanArray::from(vec![true, false]))
                as Arc<dyn arrow::array::Array>;
            let col_str = Arc::new(arrow::array::StringArray::from(vec!["Foo", "Bar"]))
                as Arc<dyn arrow::array::Array>;
            let input_batch = RecordBatch::try_new(schema, vec![col_int, col_bool, col_str])
                .expect("Failed to create record batch");

            let result = output_csv.process(input_batch);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                err.to_string()
                    .contains("Encountered unequal lengths between records on CSV file.")
            );
        }
    }
}
