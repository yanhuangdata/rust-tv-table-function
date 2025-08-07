use crate::{PathTrav, TableFunction, get_external_dir};
use anyhow::{Context, anyhow};
use arrow::{
    array::{Array, StringArray, UInt64Array, new_null_array},
    csv::Writer,
    record_batch::RecordBatch,
};
use arrow_csv::WriterBuilder;
use arrow_schema::{DataType, Field, Schema, SchemaBuilder};
use parking_lot::Mutex;
use rust_tvtf_api::arg::{Arg, Args};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Seek, Write},
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
    expected_schema: Mutex<Option<Arc<Schema>>>,
    should_write_headers: bool,
}

impl OutputCsv {
    /// Parse existing CSV header to determine expected schema
    fn parse_existing_header(
        path: &PathBuf,
        actual_first_schema: &Schema,
    ) -> anyhow::Result<Option<Arc<Schema>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut first_line = String::new();

        if reader.read_line(&mut first_line)? == 0 {
            return Ok(None); // Empty file
        }

        let mut reader = csv::Reader::from_reader(first_line.as_bytes());
        let headers = reader.headers().context("Failed to parse headers")?;

        if headers.is_empty() {
            return Ok(None);
        }
        let fields: Vec<Field> = headers
            .into_iter()
            .map(|name| {
                let data_type = actual_first_schema
                    .field_with_name(name)
                    .map(|x| x.data_type())
                    .unwrap_or(&DataType::Utf8);
                Field::new(name, data_type.clone(), true)
            })
            .collect();

        Ok(Some(Arc::new(Schema::new(fields))))
    }

    fn normalize_record_batch(
        batch: RecordBatch,
        expected_schema: &Schema,
    ) -> anyhow::Result<RecordBatch> {
        let input_schema = batch.schema();
        let mut columns = Vec::new();

        let mut input_columns: HashMap<String, Arc<dyn Array>> = HashMap::new();
        for (i, field) in input_schema.fields().iter().enumerate() {
            input_columns.insert(field.name().clone(), batch.column(i).clone());
        }
        let mut new_schema = SchemaBuilder::new();
        // Build columns according to expected schema
        for expected_field in expected_schema.fields() {
            new_schema.push(expected_field.as_ref().clone().with_nullable(true));
            let column_name = expected_field.name();
            if let Some(input_column) = input_columns.get(column_name) {
                columns.push(input_column.clone());
            } else {
                columns.push(new_null_array(expected_field.data_type(), batch.num_rows()));
            }
        }

        RecordBatch::try_new(Arc::new(expected_schema.clone()), columns)
            .context("Failed to create normalized record batch")
    }

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

        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create parent directory for file `{}`",
                    parent.to_string_lossy()
                )
            })?;
        }

        // Handle append vs create logic
        let mut file = if append {
            File::options()
                .create(true)
                .append(true)
                .read(true)
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

        let empty_file = file.metadata().map(|meta| meta.len() == 0).unwrap_or(true);
        if append && !empty_file {
            file.seek(std::io::SeekFrom::End(-1))
                .context("Failed to seek last byte")?;
            let mut buf = [0u8; 1];
            file.read_exact(&mut buf)
                .context("Failed to read last byte")?;
            if buf[0] != b'\n' {
                file.write_all(b"\n")
                    .context("failed to insert last new line")?;
            }
            file.seek(std::io::SeekFrom::Start(0))
                .context("Failed to seek first byte")?;
        }

        let should_write_headers = if append { empty_file } else { true };

        let writer = WriterBuilder::new()
            .with_header(should_write_headers)
            .with_timestamp_format("%Y-%m-%dT%H:%M:%S%.3f+00:00".to_string()) // This format is fine for the non-timezone part
            .with_timestamp_tz_format("%Y-%m-%dT%H:%M:%S%.3f%z".to_string()) // ISO 8601 with timezone
            .build(BufWriter::new(file));

        Ok(OutputCsv {
            target_path: target_path.to_path_buf(),
            writer: writer.into(),
            tee,
            append,
            rows_written: 0.into(),
            expected_schema: Mutex::new(None),
            should_write_headers,
        })
    }
}

impl TableFunction for OutputCsv {
    fn process(&mut self, input: RecordBatch) -> anyhow::Result<Option<RecordBatch>> {
        let mut expected_schema_guard = self.expected_schema.lock();

        let expected_schema = if let Some(ref schema) = *expected_schema_guard {
            schema.clone()
        } else if self.append && !self.should_write_headers {
            let parsed = Self::parse_existing_header(&self.target_path, &input.schema())?
                .context("Failed to parse header")?;
            *expected_schema_guard = Some(parsed.clone());
            parsed
        } else {
            *expected_schema_guard = Some(input.schema().clone());
            input.schema()
        };

        drop(expected_schema_guard);

        let normalized_batch = Self::normalize_record_batch(input.clone(), &expected_schema)
            .context("Failed to normalize_record_batch")?;

        let mut writer = self.writer.lock();
        writer.write(&normalized_batch)?;
        self.rows_written.fetch_add(
            normalized_batch.num_rows() as u64,
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
        let names: Vec<Option<String>> = (0..rows).map(|i| Some(format!("Name_{i}"))).collect();

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
    fn test_output_csv_append_with_schema_mismatch() {
        let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        let path = temp_file.path().to_str().unwrap().to_string();

        // Pre-populate the file with CSV content that has different schema
        {
            let mut file = File::create(&path).expect("Failed to create file");
            writeln!(file, "city,country,population,temperature").expect("Failed to write header");
            writeln!(file, "Beijing,China,2100,20.25").expect("Failed to write data");
        }

        let mut output_csv = OutputCsv::new(
            Some(vec![Arg::String(path.clone())]),
            vec![("append".to_string(), Arg::Bool(true))],
        )
        .expect("Failed to create OutputCsv instance in append mode");

        // Create a batch with fewer columns than the existing file
        let schema = Arc::new(Schema::new(vec![
            Field::new("city", DataType::Utf8, false),
            Field::new("population", DataType::Utf8, false),
        ]));
        let city = Arc::new(StringArray::from(vec!["London"])) as Arc<dyn arrow::array::Array>;
        let population = Arc::new(StringArray::from(vec!["900"])) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(schema, vec![city, population])
            .expect("Failed to create record batch");

        output_csv.process(batch).expect("Processing failed");
        output_csv.finalize().expect("Finalize failed");

        let file_content = fs::read_to_string(&path).expect("Failed to read CSV file");
        let lines: Vec<&str> = file_content.lines().collect();

        // Should maintain the original header structure and fill missing columns
        assert_eq!(lines[0], "city,country,population,temperature");
        assert_eq!(lines[1], "Beijing,China,2100,20.25");
        assert_eq!(lines[2], "London,,900,"); // Missing columns filled with empty values
    }

    #[test]
    fn test_output_csv_new_invalid_path() {
        let invalid_path = "/nonexistent_dir/invalid/file.csv";
        let result = OutputCsv::new(Some(vec![Arg::String(invalid_path.to_string())]), vec![]);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to create parent directory for file"),
            "Error message should indicate file creation failure"
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

        // First batch establishes the expected schema
        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("城市", DataType::Utf8, true),
                Field::new("国家", DataType::Utf8, true),
                Field::new("人口", DataType::Utf8, true),
                Field::new("温度", DataType::Utf8, true),
            ]));
            let city =
                Arc::new(StringArray::from(vec!["北京", "上海"])) as Arc<dyn arrow::array::Array>;
            let country =
                Arc::new(StringArray::from(vec!["中国", "中国"])) as Arc<dyn arrow::array::Array>;
            let population =
                Arc::new(StringArray::from(vec!["2100", "2500"])) as Arc<dyn arrow::array::Array>;
            let temperature =
                Arc::new(StringArray::from(vec!["20.25", "25.25"])) as Arc<dyn arrow::array::Array>;
            let input_batch =
                RecordBatch::try_new(schema, vec![city, country, population, temperature])
                    .expect("Failed to create record batch");

            let result = output_csv.process(input_batch);
            assert!(result.is_ok(), "Processing first batch should succeed");
        }

        // Second batch has fewer columns - should fill missing columns with empty strings
        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("城市", DataType::Utf8, false),
                Field::new("人口", DataType::Utf8, false),
            ]));
            let city = Arc::new(StringArray::from(vec!["London"])) as Arc<dyn arrow::array::Array>;
            let population =
                Arc::new(StringArray::from(vec!["900"])) as Arc<dyn arrow::array::Array>;
            let input_batch = RecordBatch::try_new(schema, vec![city, population])
                .expect("Failed to create record batch");

            let result = output_csv.process(input_batch);
            assert!(result.is_ok(), "Processing second batch should succeed");
        }

        // Third batch has extra columns - should drop extra columns
        {
            let schema = Arc::new(Schema::new(vec![
                Field::new("城市", DataType::Utf8, false),
                Field::new("国家", DataType::Utf8, false),
                Field::new("人口", DataType::Utf8, false),
                Field::new("温度", DataType::Utf8, false),
                Field::new("extra_column", DataType::Utf8, false), // This should be dropped
            ]));
            let city = Arc::new(StringArray::from(vec!["Tokyo"])) as Arc<dyn arrow::array::Array>;
            let country =
                Arc::new(StringArray::from(vec!["Japan"])) as Arc<dyn arrow::array::Array>;
            let population =
                Arc::new(StringArray::from(vec!["1400"])) as Arc<dyn arrow::array::Array>;
            let temperature =
                Arc::new(StringArray::from(vec!["18.5"])) as Arc<dyn arrow::array::Array>;
            let extra = Arc::new(StringArray::from(vec!["should_be_dropped"]))
                as Arc<dyn arrow::array::Array>;
            let input_batch =
                RecordBatch::try_new(schema, vec![city, country, population, temperature, extra])
                    .expect("Failed to create record batch");

            let result = output_csv.process(input_batch);
            assert!(
                result.is_ok(),
                "Processing should succeed with extra columns dropped"
            );
        }

        output_csv.finalize().expect("Finalize failed");

        let file_content = fs::read_to_string(&path).expect("Failed to read CSV file");
        println!("File content:\n{file_content}");

        // Verify the CSV structure is maintained with proper empty field handling
        let lines: Vec<&str> = file_content.lines().collect();
        assert_eq!(lines.len(), 5); // header + 4 data rows
        assert_eq!(lines[0], "城市,国家,人口,温度");
        assert_eq!(lines[1], "北京,中国,2100,20.25");
        assert_eq!(lines[2], "上海,中国,2500,25.25");
        assert_eq!(lines[3], "London,,900,"); // Missing 国家 and 温度 filled with empty strings
        assert_eq!(lines[4], "Tokyo,Japan,1400,18.5"); // Extra column dropped
    }

    #[test]
    fn test_output_csv_timestamp_formatting() {
        let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        let path = temp_file.path().to_str().unwrap().to_string();

        let mut output_csv = OutputCsv::new(Some(vec![Arg::String(path.clone())]), vec![])
            .expect("Failed to create OutputCsv instance");

        // Create a batch with various timestamp types
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "timestamp_ms",
                DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, None),
                true,
            ),
            Field::new(
                "timestamp_us",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
                true,
            ),
            Field::new(
                "timestamp_ns",
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                true,
            ),
        ]));

        let ids =
            Arc::new(arrow::array::Int32Array::from(vec![1, 2, 3])) as Arc<dyn arrow::array::Array>;

        // Create timestamp arrays with known values
        let timestamp_ms = Arc::new(arrow::array::TimestampMillisecondArray::from(vec![
            Some(1609459200000), // 2021-01-01 00:00:00.000
            Some(1609459260123), // 2021-01-01 00:01:00.123
            None,
        ])) as Arc<dyn arrow::array::Array>;

        let timestamp_us = Arc::new(arrow::array::TimestampMicrosecondArray::from(vec![
            Some(1609459200000000), // 2021-01-01 00:00:00.000000
            Some(1609459260123456), // 2021-01-01 00:01:00.123456
            None,
        ])) as Arc<dyn arrow::array::Array>;

        let timestamp_ns = Arc::new(arrow::array::TimestampNanosecondArray::from(vec![
            Some(1609459200000000000), // 2021-01-01 00:00:00.000000000
            Some(1609459260123456789), // 2021-01-01 00:01:00.123456789
            None,
        ])) as Arc<dyn arrow::array::Array>;

        let batch =
            RecordBatch::try_new(schema, vec![ids, timestamp_ms, timestamp_us, timestamp_ns])
                .expect("Failed to create record batch");

        output_csv.process(batch).expect("Processing failed");
        output_csv.finalize().expect("Finalize failed");

        let file_content = fs::read_to_string(&path).expect("Failed to read CSV file");
        println!("Timestamp CSV content:\n{file_content}");

        let lines: Vec<&str> = file_content.lines().collect();
        assert_eq!(lines.len(), 4); // header + 3 data rows
        assert_eq!(lines[0], "id,timestamp_ms,timestamp_us,timestamp_ns");

        // Verify timestamp formatting (should use custom format: %Y-%m-%dT%H:%M:%S%.3f)
        assert!(lines[1].contains("2021-01-01T00:00:00"));
        assert!(lines[2].contains("2021-01-01T00:01:00"));
        assert!(lines[3].contains(",,,")); // null values
    }

    #[test]
    fn test_output_csv_timestamp_with_timezone_formatting() {
        let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        let path = temp_file.path().to_str().unwrap().to_string();

        let mut output_csv = OutputCsv::new(Some(vec![Arg::String(path.clone())]), vec![])
            .expect("Failed to create OutputCsv instance");

        // Create a batch with timezone-aware timestamps
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "timestamp_utc",
                DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, Some("+00:00".into())),
                true,
            ),
            Field::new(
                "timestamp_est",
                DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, Some("-05:00".into())),
                true,
            ),
            Field::new(
                "timestamp_jst",
                DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, Some("+09:00".into())),
                true,
            ),
        ]));

        let ids =
            Arc::new(arrow::array::Int32Array::from(vec![1, 2])) as Arc<dyn arrow::array::Array>;

        // Create timezone-aware timestamp arrays
        let timestamp_utc = Arc::new(
            arrow::array::TimestampMillisecondArray::from(vec![
                Some(1609459200000), // 2021-01-01 00:00:00.000 UTC
                Some(1609459260123), // 2021-01-01 00:01:00.123 UTC
            ])
            .with_timezone("+00:00".to_string()),
        ) as Arc<dyn arrow::array::Array>;

        let timestamp_est = Arc::new(
            arrow::array::TimestampMillisecondArray::from(vec![
                Some(1609459200000), // Same UTC time, different timezone
                Some(1609459260123),
            ])
            .with_timezone("-05:00".to_string()),
        ) as Arc<dyn arrow::array::Array>;

        let timestamp_jst = Arc::new(
            arrow::array::TimestampMillisecondArray::from(vec![
                Some(1609459200000), // Same UTC time, different timezone
                Some(1609459260123),
            ])
            .with_timezone("+09:00".to_string()),
        ) as Arc<dyn arrow::array::Array>;

        let batch = RecordBatch::try_new(
            schema,
            vec![ids, timestamp_utc, timestamp_est, timestamp_jst],
        )
        .expect("Failed to create record batch");

        output_csv.process(batch).expect("Processing failed");
        output_csv.finalize().expect("Finalize failed");

        let file_content = fs::read_to_string(&path).expect("Failed to read CSV file");
        println!("Timezone CSV content:\n{file_content}");

        let lines: Vec<&str> = file_content.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 data rows
        assert_eq!(lines[0], "id,timestamp_utc,timestamp_est,timestamp_jst");

        // Verify timezone formatting (should use custom format: %Y-%m-%d %H:%M:%S%.3f %z)
        assert!(lines[1].contains("+0000") || lines[1].contains("+00:00"));
        assert!(lines[1].contains("-0500") || lines[1].contains("-05:00"));
        assert!(lines[1].contains("+0900") || lines[1].contains("+09:00"));
    }

    #[test]
    fn test_output_csv_mixed_timestamp_types() {
        let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        let path = temp_file.path().to_str().unwrap().to_string();

        let mut output_csv = OutputCsv::new(Some(vec![Arg::String(path.clone())]), vec![])
            .expect("Failed to create OutputCsv instance");

        // Create a batch mixing regular timestamps and timezone-aware timestamps
        let schema = Arc::new(Schema::new(vec![
            Field::new("event_id", DataType::Utf8, false),
            Field::new(
                "local_time",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                true,
            ),
            Field::new(
                "utc_time",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, Some("UTC".into())),
                true,
            ),
            Field::new("date_only", DataType::Date32, true),
            Field::new(
                "time_only",
                DataType::Time32(arrow_schema::TimeUnit::Second),
                true,
            ),
        ]));

        let event_ids = Arc::new(StringArray::from(vec!["event1", "event2", "event3"]))
            as Arc<dyn arrow::array::Array>;

        let local_time = Arc::new(arrow::array::TimestampSecondArray::from(vec![
            Some(1609459200), // 2021-01-01 00:00:00
            Some(1609545600), // 2021-01-02 00:00:00
            None,
        ])) as Arc<dyn arrow::array::Array>;

        let utc_time = Arc::new(
            arrow::array::TimestampSecondArray::from(vec![
                Some(1609459200), // 2021-01-01 00:00:00 UTC
                Some(1609545600), // 2021-01-02 00:00:00 UTC
                Some(1609632000), // 2021-01-03 00:00:00 UTC
            ])
            .with_timezone("UTC".to_string()),
        ) as Arc<dyn arrow::array::Array>;

        let date_only = Arc::new(arrow::array::Date32Array::from(vec![
            Some(18628), // 2021-01-01
            Some(18629), // 2021-01-02
            None,
        ])) as Arc<dyn arrow::array::Array>;

        let time_only = Arc::new(arrow::array::Time32SecondArray::from(vec![
            Some(0),     // 00:00:00
            Some(3661),  // 01:01:01
            Some(86399), // 23:59:59
        ])) as Arc<dyn arrow::array::Array>;

        let batch = RecordBatch::try_new(
            schema,
            vec![event_ids, local_time, utc_time, date_only, time_only],
        )
        .expect("Failed to create record batch");

        output_csv.process(batch).expect("Processing failed");
        output_csv.finalize().expect("Finalize failed");

        let file_content = fs::read_to_string(&path).expect("Failed to read CSV file");
        println!("Mixed timestamp CSV content:\n{file_content}");

        let lines: Vec<&str> = file_content.lines().collect();
        assert_eq!(lines.len(), 4); // header + 3 data rows
        assert_eq!(lines[0], "event_id,local_time,utc_time,date_only,time_only");

        // Verify that different timestamp types are handled correctly
        assert!(lines[1].contains("event1"));
        assert!(lines[2].contains("event2"));
        assert!(lines[3].contains("event3"));

        // Check that dates and times are formatted properly
        assert!(lines[1].contains("2021-01-01") || lines[1].contains("1970-01-01")); // Date formatting
        assert!(lines[1].contains("00:00:00")); // Time formatting
    }
}
