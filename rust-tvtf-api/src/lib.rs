use anyhow::Context;
use arg::Args;
use arrow::array::RecordBatch;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow_utils::{DynamicArrowArrayStreamReader, VecRecordBatchReader};
use derive_builder::Builder;
use serde::Serialize;
use std::ffi::c_char;
use std::ptr::null_mut;
use std::sync::Arc;

use crate::arg::ArgType;

pub mod arg;
mod arrow_utils;

/// # SAFETY
///
/// This function is unsafe because it dereferences raw pointers.
///
/// For `parameters`, it expects a null-terminated UTF-8 string, it may be `nullptr`.
///
/// For `timezone`, it expects a null-terminated UTF-8 string, it must be valid.
pub unsafe fn create_raw(
    registry: &FunctionRegistry,
    parameters: *const i8,
    timezone: *const i8,
) -> anyhow::Result<Box<dyn TableFunction>> {
    let parameters = if parameters.is_null() {
        None
    } else {
        unsafe {
            Some(std::str::from_utf8_unchecked(
                std::ffi::CStr::from_ptr(parameters as *const c_char).to_bytes(),
            ))
        }
    };
    let timezone = unsafe {
        std::str::from_utf8_unchecked(
            std::ffi::CStr::from_ptr(timezone as *const c_char).to_bytes(),
        )
    };
    create(registry, parameters, timezone)
}

pub fn create(
    registry: &FunctionRegistry,
    parameters: Option<&str>,
    timezone: &str,
) -> anyhow::Result<Box<dyn TableFunction>> {
    let create_closure = &(registry.init);
    let parameters = if let Some(param) = parameters {
        serde_json::from_str(param).context("serde json failed")?
    } else {
        None
    };
    let ctx = FunctionContext {
        parameters,
        // TODO: parse as value instead of string
        timezone: String::from(timezone),
    };
    create_closure(ctx)
}

type TableFunctionInitialize =
    Arc<dyn Fn(FunctionContext) -> anyhow::Result<Box<dyn TableFunction>>>;

#[derive(Builder)]
pub struct FunctionRegistry {
    #[builder(setter(into))]
    name: &'static str,
    init: TableFunctionInitialize,
    #[builder(setter(strip_option, each(name = "signature", into)))]
    signatures: Option<Vec<Signature>>,
}

impl std::fmt::Debug for FunctionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionRegistry")
            .field("name", &self.name)
            .field("init", &Arc::as_ptr(&self.init))
            .field("signatures", &self.signatures)
            .finish()
    }
}

impl FunctionRegistry {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn signatures(&self) -> anyhow::Result<String> {
        serde_json::to_string(&self.signatures).context("Failed to get signatures")
    }

    pub fn builder() -> FunctionRegistryBuilder {
        FunctionRegistryBuilder::default()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Signature {
    pub args: Vec<ArgType>,
}

impl From<Vec<ArgType>> for Signature {
    fn from(value: Vec<ArgType>) -> Self {
        Signature { args: value }
    }
}

pub struct FunctionContext {
    pub parameters: Option<Args>,
    pub timezone: String,
}

pub trait TableFunction {
    fn process(&mut self, input: RecordBatch) -> anyhow::Result<Option<RecordBatch>>;

    fn finalize(&mut self) -> anyhow::Result<Option<RecordBatch>> {
        Ok(None)
    }
}

/// Wrapper over `process` method, the type of `input_stream` is `*mut FFI_ArrowArrayStream`.
/// Due to the limitation of zngur, use `i64` here as `void*` and casting to `*mut FFI_ArrowArrayStream`.
///
/// The input_stream should contain 0 or 1 RecordBatch
///
/// Returns may be `nullptr`. Otherwise, returns `*mut FFI_ArrowArrayStream`
///
/// # SAFETY
///
/// This function is unsafe because it dereferences a raw pointer and
/// expects the caller to ensure that the pointer is valid and
/// points to a `Box<dyn TableFunction>`.
pub unsafe fn process_raw(
    func: &mut Box<dyn TableFunction>,
    input_stream: i64,
) -> anyhow::Result<i64> {
    let mut stream_reader: DynamicArrowArrayStreamReader = unsafe {
        DynamicArrowArrayStreamReader::from_raw(
            input_stream as *mut arrow::ffi_stream::FFI_ArrowArrayStream,
        )
        .expect("Failed to construct DynamicArrowArrayStreamReader")
    };
    if let Some(record_batch) = stream_reader.next() {
        let record_batch = record_batch.expect("cannot iterate over record batch");
        let Some(output) = func.process(record_batch)? else {
            return Ok(null_mut::<FFI_ArrowArrayStream>() as i64);
        };
        let boxed = Box::new(FFI_ArrowArrayStream::new(VecRecordBatchReader::new(vec![
            output,
        ])));
        return Ok(Box::into_raw(boxed) as i64);
    }

    Ok(null_mut::<FFI_ArrowArrayStream>() as i64)
}

/// Wrapper over `finalize` method.
///
/// Returns may be `nullptr`. Otherwise, returns `i64` as `*mut FFI_ArrowArrayStream`.
///
/// # SAFETY
///
/// This function is unsafe because it dereferences a raw pointer and
/// expects the caller to ensure that the pointer is valid and
/// points to a `Box<dyn TableFunction>`.
pub unsafe fn finalize_raw(func: &mut Box<dyn TableFunction>) -> anyhow::Result<i64> {
    let Some(output) = func.finalize()? else {
        return Ok(null_mut::<FFI_ArrowArrayStream>() as i64);
    };
    let boxed = Box::new(FFI_ArrowArrayStream::new(VecRecordBatchReader::new(vec![
        output,
    ])));
    Ok(Box::into_raw(boxed) as i64)
}
