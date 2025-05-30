// copied from arrow/ffi_stream.rs, but make it dynamic schema reader
use std::convert::TryFrom;
use std::{ffi::CStr, sync::Arc};

use arrow::array::{RecordBatchReader, StructArray};
use arrow::ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::RecordBatch;
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};

#[derive(Debug)]
pub struct DynamicArrowArrayStreamReader {
    stream: FFI_ArrowArrayStream,
    schema: SchemaRef,
}
/// Gets schema from a raw pointer of `FFI_ArrowArrayStream`. This is used when constructing
/// `ArrowArrayStreamReader` to cache schema.
fn get_stream_schema(stream_ptr: *mut FFI_ArrowArrayStream) -> Result<SchemaRef, ArrowError> {
    let mut schema = FFI_ArrowSchema::empty();

    let ret_code = unsafe { (*stream_ptr).get_schema.unwrap()(stream_ptr, &mut schema) };

    if ret_code == 0 {
        let schema = Schema::try_from(&schema)?;
        Ok(Arc::new(schema))
    } else {
        Err(ArrowError::CDataInterface(format!(
            "Cannot get schema from input stream. Error code: {ret_code:?}"
        )))
    }
}

impl DynamicArrowArrayStreamReader {
    /// Creates a new `ArrowArrayStreamReader` from a `FFI_ArrowArrayStream`.
    /// This is used to import from the C Stream Interface.
    #[allow(dead_code)]
    pub fn try_new(mut stream: FFI_ArrowArrayStream) -> Result<Self, ArrowError> {
        if stream.release.is_none() {
            return Err(ArrowError::CDataInterface(
                "input stream is already released".to_string(),
            ));
        }

        let schema = get_stream_schema(&mut stream)?;

        Ok(Self { stream, schema })
    }

    /// Creates a new `ArrowArrayStreamReader` from a raw pointer of `FFI_ArrowArrayStream`.
    ///
    /// Assumes that the pointer represents valid C Stream Interfaces.
    /// This function copies the content from the raw pointer and cleans up it to prevent
    /// double-dropping. The caller is responsible for freeing up the memory allocated for
    /// the pointer.
    ///
    /// # Safety
    ///
    /// See [`FFI_ArrowArrayStream::from_raw`]
    #[allow(dead_code)]
    pub unsafe fn from_raw(raw_stream: *mut FFI_ArrowArrayStream) -> Result<Self, ArrowError> {
        Self::try_new(FFI_ArrowArrayStream::from_raw(raw_stream))
    }

    /// Get the last error from `ArrowArrayStreamReader`
    fn get_stream_last_error(&mut self) -> Option<String> {
        let get_last_error = self.stream.get_last_error?;

        let error_str = unsafe { get_last_error(&mut self.stream) };
        if error_str.is_null() {
            return None;
        }

        let error_str = unsafe { CStr::from_ptr(error_str) };
        Some(error_str.to_string_lossy().to_string())
    }
}

impl Iterator for DynamicArrowArrayStreamReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        // main change is here
        match get_stream_schema(&mut self.stream) {
            Ok(schema) => self.schema = schema,
            Err(e) => return Some(Err(e)),
        }
        let mut array = FFI_ArrowArray::empty();

        let ret_code = unsafe { self.stream.get_next.unwrap()(&mut self.stream, &mut array) };

        if ret_code == 0 {
            // The end of stream has been reached
            if array.is_released() {
                return None;
            }

            let result = unsafe {
                from_ffi_and_data_type(array, DataType::Struct(self.schema().fields().clone()))
            };
            let batch_with_schema_result = result.map(|data| {
                RecordBatch::from(StructArray::from(data)).with_schema(self.schema.clone())
            });
            match batch_with_schema_result {
                Ok(batch) => Some(batch),
                Err(e) => Some(Err(e)),
            }
        } else {
            let last_error = self.get_stream_last_error();
            let err = ArrowError::CDataInterface(last_error.unwrap());
            Some(Err(err))
        }
    }
}

impl RecordBatchReader for DynamicArrowArrayStreamReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

pub struct VecRecordBatchReader {
    schema: SchemaRef,
    iter: Box<dyn Iterator<Item = RecordBatch> + Send>,
}

impl VecRecordBatchReader {
    pub fn new(vec: Vec<RecordBatch>) -> Box<VecRecordBatchReader> {
        let schema = match vec.len() {
            0 => Arc::new(Schema::empty()),
            _ => vec[0].schema(),
        };
        let iter = Box::new(vec.into_iter());
        Box::new(VecRecordBatchReader { schema, iter })
    }
}

impl Iterator for VecRecordBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(Ok)
    }
}

impl RecordBatchReader for VecRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;

    use super::*;

    use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};

    struct TestRecordBatchReader {
        batches: Vec<RecordBatch>,
        position: usize,
    }

    impl TestRecordBatchReader {
        pub fn new(batches: Vec<RecordBatch>) -> Box<TestRecordBatchReader> {
            Box::new(TestRecordBatchReader {
                batches,
                position: 0,
            })
        }
    }

    impl Iterator for TestRecordBatchReader {
        type Item = Result<RecordBatch, ArrowError>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.position < self.batches.len() {
                self.position += 1;
                Some(Ok(self.batches[self.position - 1].clone()))
            } else {
                None
            }
        }
    }

    impl RecordBatchReader for TestRecordBatchReader {
        fn schema(&self) -> SchemaRef {
            if self.position < self.batches.len() {
                self.batches[self.position].schema()
            } else {
                Arc::new(Schema::empty())
            }
        }
    }

    #[test]
    fn test_dynamic_schema_stream_round_trip_export() {
        let schema1 = Arc::new(Schema::new(vec![
            Field::new("f1", DataType::Int64, true),
            Field::new("f2", DataType::Utf8, true),
            Field::new("f3", DataType::Boolean, true),
        ]));
        let batch1 = RecordBatch::try_new(
            schema1.clone(),
            vec![
                Arc::new(Int64Array::from(vec![Some(2), None, Some(1), None])),
                Arc::new(StringArray::from(vec![
                    None,
                    Some("b"),
                    Some("c"),
                    Some("d"),
                ])),
                Arc::new(BooleanArray::from(vec![
                    Some(true),
                    None,
                    Some(true),
                    Some(false),
                ])),
            ],
        )
        .unwrap();

        let schema2 = Arc::new(Schema::new(vec![
            Field::new("f4", DataType::Float64, true),
            Field::new("f2", DataType::Utf8, true),
        ]));
        let batch2 = RecordBatch::try_new(
            schema2,
            vec![
                Arc::new(Float64Array::from(vec![Some(2.2), None, None])),
                Arc::new(StringArray::from(vec![None, Some("bd"), Some("cef")])),
            ],
        )
        .unwrap();

        let batches = vec![batch1.clone(), batch2.clone()];
        let reader = TestRecordBatchReader::new(batches);

        // Import through `FFI_ArrowArrayStream` as `ArrowArrayStreamReader`
        let stream = FFI_ArrowArrayStream::new(reader);
        let stream_reader = DynamicArrowArrayStreamReader::try_new(stream).unwrap();

        let imported_schema = stream_reader.schema();
        assert_eq!(imported_schema, schema1.clone());

        let mut produced_batches = vec![];
        for batch in stream_reader {
            produced_batches.push(batch.unwrap());
        }

        assert_eq!(produced_batches, vec![batch1.clone(), batch2.clone()]);
    }
}
