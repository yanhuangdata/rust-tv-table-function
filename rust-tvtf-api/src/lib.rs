use crate::arg::{Arg, ArgType, NamedArg};
use anyhow::Context;
use arg::Args;
use arrow::array::RecordBatch;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow_utils::{DynamicArrowArrayStreamReader, VecRecordBatchReader};
use derive_builder::Builder;
use serde::Serialize;
use std::borrow::Cow;
use std::ffi::c_char;
use std::ptr::null_mut;
use std::sync::Arc;

pub mod arg;
mod arrow_utils;

/// Log macro for FFI entry points with location info
macro_rules! ffi_log {
    (info, $($arg:tt)*) => {
        log::info!("[FFI-API] {}", format!($($arg)*))
    };
    (debug, $($arg:tt)*) => {
        log::debug!("[FFI-API] {}", format!($($arg)*))
    };
    (error, $($arg:tt)*) => {
        log::error!("[FFI-API] {}", format!($($arg)*))
    };
    (warn, $($arg:tt)*) => {
        log::warn!("[FFI-API] {}", format!($($arg)*))
    };
}

/// # SAFETY
///
/// This function is unsafe because it dereferences raw pointers.
///
/// For `arguments`, it expects a null-terminated UTF-8 string, it may be `nullptr`.
///
/// For `named_arguments`, it expects a null-terminated UTF-8 string, it may be `nullptr`.
///
/// For `timezone`, it expects a null-terminated UTF-8 string, it must be valid.
pub unsafe fn create_raw(
    registry: &FunctionRegistry,
    arguments: *const i8,
    named_arguments: *const i8,
    timezone: *const i8,
) -> anyhow::Result<Box<dyn TableFunction>> {
    ffi_log!(info, "=== create_raw() ENTRY ===");
    ffi_log!(debug, "create_raw: registry.name={}", registry.name());
    ffi_log!(
        debug,
        "create_raw: arguments ptr={:?}, named_arguments ptr={:?}, timezone ptr={:?}",
        arguments,
        named_arguments,
        timezone
    );

    let arguments = if arguments.is_null() {
        ffi_log!(debug, "create_raw: arguments is null");
        None
    } else {
        unsafe {
            let args_str = std::str::from_utf8_unchecked(
                std::ffi::CStr::from_ptr(arguments as *const c_char).to_bytes(),
            );
            ffi_log!(debug, "create_raw: arguments={}", args_str);
            Some(args_str)
        }
    };
    let named_arguments = if named_arguments.is_null() {
        ffi_log!(debug, "create_raw: named_arguments is null");
        None
    } else {
        unsafe {
            let named_str = std::str::from_utf8_unchecked(
                std::ffi::CStr::from_ptr(named_arguments as *const c_char).to_bytes(),
            );
            ffi_log!(debug, "create_raw: named_arguments={}", named_str);
            Some(named_str)
        }
    };
    let timezone = unsafe {
        let tz_str = std::str::from_utf8_unchecked(
            std::ffi::CStr::from_ptr(timezone as *const c_char).to_bytes(),
        );
        ffi_log!(debug, "create_raw: timezone={}", tz_str);
        tz_str
    };

    ffi_log!(debug, "create_raw: calling create()...");
    let result = create(registry, arguments, named_arguments, timezone);

    match &result {
        Ok(_) => ffi_log!(info, "create_raw: SUCCESS - TableFunction created"),
        Err(e) => ffi_log!(error, "create_raw: FAILED - {:?}", e),
    }

    ffi_log!(info, "=== create_raw() EXIT ===");
    result
}

pub fn create(
    registry: &FunctionRegistry,
    arguments: Option<&str>,
    named_arguments: Option<&str>,
    timezone: &str,
) -> anyhow::Result<Box<dyn TableFunction>> {
    ffi_log!(debug, "create: Parsing arguments...");

    let create_closure = &(registry.init);
    let arguments = if let Some(arg) = arguments {
        ffi_log!(debug, "create: Parsing positional arguments JSON: {}", arg);
        let parsed: Option<Args> =
            serde_json::from_str(arg).context("Failed to parse arguments from JSON")?;
        ffi_log!(debug, "create: Parsed positional arguments: {:?}", parsed);
        parsed
    } else {
        ffi_log!(debug, "create: No positional arguments");
        None
    };
    let named_arguments: Vec<NamedArg> = if let Some(arg) = named_arguments {
        ffi_log!(debug, "create: Parsing named arguments JSON: {}", arg);
        let parsed: Vec<NamedArg> =
            serde_json::from_str(arg).context("Failed to parse named arguments from JSON")?;
        ffi_log!(debug, "create: Parsed {} named arguments", parsed.len());
        for (i, na) in parsed.iter().enumerate() {
            ffi_log!(
                debug,
                "create:   NamedArg[{}]: name={}, arg={:?}",
                i,
                na.name,
                na.arg
            );
        }
        parsed
    } else {
        ffi_log!(debug, "create: No named arguments");
        vec![]
    };

    ffi_log!(
        debug,
        "create: Building FunctionContext with timezone={}",
        timezone
    );
    let ctx = FunctionContext {
        arguments,
        named_arguments: named_arguments
            .into_iter()
            .map(|named| (named.name, named.arg))
            .collect(),
        // TODO: parse as value instead of string
        timezone: String::from(timezone),
    };

    ffi_log!(
        debug,
        "create: Calling init closure for function '{}'...",
        registry.name()
    );
    let result = create_closure(ctx);

    match &result {
        Ok(_) => ffi_log!(debug, "create: init closure succeeded"),
        Err(e) => ffi_log!(error, "create: init closure failed: {:?}", e),
    }

    result
}

type TableFunctionInitialize =
    Arc<dyn Fn(FunctionContext) -> anyhow::Result<Box<dyn TableFunction>>>;

#[derive(Builder, Clone)]
pub struct FunctionRegistry {
    #[builder(setter(into))]
    name: &'static str,
    init: TableFunctionInitialize,
    #[builder(setter(strip_option, each(name = "signature", into)))]
    signatures: Option<Vec<Signature>>,
    #[builder(default = false)]
    require_ordered: bool,
}

impl std::fmt::Debug for FunctionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionRegistry")
            .field("name", &self.name)
            .field("init", &Arc::as_ptr(&self.init))
            .field("signatures", &self.signatures)
            .field("require_ordered", &self.require_ordered)
            .finish()
    }
}

impl FunctionRegistry {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn require_ordered(&self) -> bool {
        self.require_ordered
    }

    pub fn signatures(&self) -> anyhow::Result<String> {
        serde_json::to_string(&self.signatures).context("Failed to get signatures")
    }

    pub fn builder() -> FunctionRegistryBuilder {
        FunctionRegistryBuilder::default()
    }
}

#[derive(Clone, Debug, Serialize, Builder)]
pub struct Signature {
    #[builder(setter(each(name = "parameter", into)))]
    pub(crate) parameters: Vec<Parameter>,
}

impl Signature {
    pub fn builder() -> SignatureBuilder {
        SignatureBuilder::default()
    }

    pub fn empty() -> Signature {
        Signature { parameters: vec![] }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Parameter {
    pub(crate) name: Option<String>,
    pub(crate) default: Option<Arg>,
    pub(crate) arg_type: ArgType,
}

impl From<ArgType> for Parameter {
    fn from(value: ArgType) -> Self {
        Parameter {
            name: None,
            default: None,
            arg_type: value,
        }
    }
}

impl<NAME, ARG> From<(Option<NAME>, ArgType, Option<ARG>)> for Parameter
where
    NAME: Into<Cow<'static, str>>,
    ARG: Into<Arg>,
{
    fn from((name, arg_type, default): (Option<NAME>, ArgType, Option<ARG>)) -> Self {
        Parameter {
            name: name.map(|x| x.into().into_owned()),
            default: default.map(|x| x.into()),
            arg_type,
        }
    }
}

impl<P> From<Vec<P>> for Signature
where
    P: Into<Parameter>,
{
    fn from(value: Vec<P>) -> Self {
        Signature {
            parameters: value.into_iter().map(|x| x.into()).collect(),
        }
    }
}

pub struct FunctionContext {
    pub arguments: Option<Args>,
    pub named_arguments: Vec<(String, Arg)>,
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
    ffi_log!(info, "=== process_raw() ENTRY ===");
    ffi_log!(debug, "process_raw: input_stream ptr=0x{:x}", input_stream);

    ffi_log!(
        debug,
        "process_raw: Creating DynamicArrowArrayStreamReader from raw ptr..."
    );
    let mut stream_reader: DynamicArrowArrayStreamReader = unsafe {
        match DynamicArrowArrayStreamReader::from_raw(
            input_stream as *mut arrow::ffi_stream::FFI_ArrowArrayStream,
        ) {
            Ok(reader) => {
                ffi_log!(
                    debug,
                    "process_raw: DynamicArrowArrayStreamReader created successfully"
                );
                reader
            }
            Err(e) => {
                ffi_log!(
                    error,
                    "process_raw: Failed to create DynamicArrowArrayStreamReader: {:?}",
                    e
                );
                panic!("Failed to construct DynamicArrowArrayStreamReader: {:?}", e);
            }
        }
    };

    ffi_log!(
        debug,
        "process_raw: Attempting to read RecordBatch from stream..."
    );
    if let Some(record_batch_result) = stream_reader.next() {
        ffi_log!(debug, "process_raw: Got record batch result from stream");

        let record_batch = match record_batch_result {
            Ok(batch) => {
                ffi_log!(
                    info,
                    "process_raw: RecordBatch received - {} rows, {} columns",
                    batch.num_rows(),
                    batch.num_columns()
                );

                // Log schema details
                let schema = batch.schema();
                ffi_log!(debug, "process_raw: RecordBatch schema:");
                for (i, field) in schema.fields().iter().enumerate() {
                    ffi_log!(
                        debug,
                        "process_raw:   Field[{}]: name={}, type={:?}, nullable={}",
                        i,
                        field.name(),
                        field.data_type(),
                        field.is_nullable()
                    );
                }

                // Log first few rows as sample
                if batch.num_rows() > 0 {
                    ffi_log!(debug, "process_raw: First row sample (up to 5 columns):");
                    for col_idx in 0..std::cmp::min(5, batch.num_columns()) {
                        let col = batch.column(col_idx);
                        ffi_log!(
                            debug,
                            "process_raw:   Column[{}] ({}) len={}, null_count={}",
                            col_idx,
                            schema.field(col_idx).name(),
                            col.len(),
                            col.null_count()
                        );
                    }
                }

                batch
            }
            Err(e) => {
                ffi_log!(error, "process_raw: Failed to read RecordBatch: {:?}", e);
                panic!("cannot iterate over record batch: {:?}", e);
            }
        };

        ffi_log!(
            debug,
            "process_raw: Calling func.process() with {} rows...",
            record_batch.num_rows()
        );
        let process_start = std::time::Instant::now();

        let process_result = func.process(record_batch);

        let process_time = process_start.elapsed();
        ffi_log!(
            debug,
            "process_raw: func.process() completed in {:?}",
            process_time
        );

        match process_result {
            Ok(Some(output)) => {
                ffi_log!(
                    info,
                    "process_raw: Output RecordBatch - {} rows, {} columns",
                    output.num_rows(),
                    output.num_columns()
                );
                let boxed = Box::new(FFI_ArrowArrayStream::new(VecRecordBatchReader::new(vec![
                    output,
                ])));
                let ptr = Box::into_raw(boxed) as i64;
                ffi_log!(
                    debug,
                    "process_raw: Created output FFI_ArrowArrayStream at 0x{:x}",
                    ptr
                );
                ffi_log!(info, "=== process_raw() EXIT (with output) ===");
                return Ok(ptr);
            }
            Ok(None) => {
                ffi_log!(
                    info,
                    "process_raw: func.process() returned None (no output)"
                );
                ffi_log!(info, "=== process_raw() EXIT (no output) ===");
                return Ok(null_mut::<FFI_ArrowArrayStream>() as i64);
            }
            Err(e) => {
                ffi_log!(error, "process_raw: func.process() FAILED: {:?}", e);
                ffi_log!(info, "=== process_raw() EXIT (error) ===");
                return Err(e);
            }
        }
    }

    ffi_log!(
        info,
        "process_raw: No RecordBatch in input stream (empty input)"
    );
    ffi_log!(info, "=== process_raw() EXIT (empty input) ===");
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
    ffi_log!(info, "=== finalize_raw() ENTRY ===");

    ffi_log!(debug, "finalize_raw: Calling func.finalize()...");
    let finalize_result = func.finalize();

    match finalize_result {
        Ok(Some(output)) => {
            ffi_log!(
                info,
                "finalize_raw: Output RecordBatch - {} rows, {} columns",
                output.num_rows(),
                output.num_columns()
            );
            let boxed = Box::new(FFI_ArrowArrayStream::new(VecRecordBatchReader::new(vec![
                output,
            ])));
            let ptr = Box::into_raw(boxed) as i64;
            ffi_log!(
                debug,
                "finalize_raw: Created output FFI_ArrowArrayStream at 0x{:x}",
                ptr
            );
            ffi_log!(info, "=== finalize_raw() EXIT (with output) ===");
            Ok(ptr)
        }
        Ok(None) => {
            ffi_log!(
                info,
                "finalize_raw: func.finalize() returned None (no output)"
            );
            ffi_log!(info, "=== finalize_raw() EXIT (no output) ===");
            Ok(null_mut::<FFI_ArrowArrayStream>() as i64)
        }
        Err(e) => {
            ffi_log!(error, "finalize_raw: func.finalize() FAILED: {:?}", e);
            ffi_log!(info, "=== finalize_raw() EXIT (error) ===");
            Err(e)
        }
    }
}

pub fn anyhow_error_to_string(error: &anyhow::Error) -> String {
    format!("{error:?}")
}
