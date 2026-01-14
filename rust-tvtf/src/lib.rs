use crate::funcs::*;
use anyhow::Context;
use rust_tvtf_api::Signature;
use rust_tvtf_api::{FunctionRegistry, TableFunction, arg::ArgType};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::{sync::Arc, vec};

pub mod funcs;
#[rustfmt::skip]
#[allow(clippy::all)]
mod zngur_generated;

pub fn get_function_registries() -> anyhow::Result<Vec<FunctionRegistry>> {
    Ok(vec![
        FunctionRegistry::builder()
            .name("addtotals")
            .init(Arc::new(|ctx| {
                AddTotals::new(ctx.arguments).map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .signature(Signature::empty())
            .signature(vec![ArgType::Int])
            .build()
            .context("create `addtotals` registry failed")?,
        FunctionRegistry::builder()
            .name("output_csv")
            .init(Arc::new(|ctx| {
                OutputCsv::new(ctx.arguments, ctx.named_arguments)
                    .map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .signature(
                Signature::builder()
                    .parameter(ArgType::String) // output path
                    .parameter((Some("append"), ArgType::Bool, Some(false)))
                    .parameter((Some("tee"), ArgType::Bool, Some(false)))
                    .build()
                    .context("Failed to build signature parameters")?,
            )
            .build()
            .context("create `output_csv` registry failed")?,
        FunctionRegistry::builder()
            .name("transaction")
            .init(Arc::new(|ctx| {
                TransFunction::new(ctx.arguments, ctx.named_arguments)
                    .map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .require_ordered(true)
            .signature(
                Signature::builder()
                    .parameter(ArgType::String) // fields
                    .parameter((Some("starts_with"), ArgType::String, Some("")))
                    .parameter((Some("starts_with_regex"), ArgType::String, Some("")))
                    .parameter((Some("starts_if"), ArgType::String, Some("")))
                    .parameter((Some("ends_with"), ArgType::String, Some("")))
                    .parameter((Some("ends_with_regex"), ArgType::String, Some("")))
                    .parameter((Some("ends_if"), ArgType::String, Some("")))
                    .parameter((Some("max_span"), ArgType::String, Some("")))
                    .parameter((Some("max_events"), ArgType::Int, Some(1000)))
                    .build()
                    .context("Failed to build signature parameters")?,
            )
            .build()
            .context("create `transaction` registry failed")?,
        FunctionRegistry::builder()
            .name("flatten")
            .init(Arc::new(|ctx| {
                Flatten::new(ctx.arguments).map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .signature(
                Signature::builder()
                    .parameter(ArgType::String) // column names (comma-separated)
                    .build()
                    .context("Failed to build signature parameters")?,
            )
            .build()
            .context("create `flatten` registry failed")?,
        FunctionRegistry::builder()
            .name("filldown")
            .init(Arc::new(|ctx| {
                Filldown::new(ctx.arguments).map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .require_ordered(true)
            .signature(Signature::empty())
            .signature(
                Signature::builder()
                    .parameter(ArgType::String) // column names (comma-separated)
                    .build()
                    .context("Failed to build signature parameters")?,
            )
            .build()
            .context("create `filldown` registry failed")?,
        FunctionRegistry::builder()
            .name("predict")
            .init(Arc::new(|ctx| {
                Predict::new(ctx.arguments, ctx.named_arguments)
                    .map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .require_ordered(true)
            .signature(
                Signature::builder()
                    .parameter(ArgType::String) // field name(s)
                    .parameter((Some("algorithm"), ArgType::String, Some("LLP5")))
                    .parameter((Some("period"), ArgType::Int, Some(0)))
                    .parameter((Some("future_timespan"), ArgType::Int, Some(10)))
                    .parameter((Some("holdback"), ArgType::Int, Some(0)))
                    .parameter((Some("upper"), ArgType::Float, Some(0.99)))
                    .parameter((Some("lower"), ArgType::Float, Some(0.99)))
                    .build()
                    .context("Failed to build signature parameters")?,
            )
            .build()
            .context("create `predict` registry failed")?,
    ])
}

pub fn get_external_dir() -> Option<PathBuf> {
    let home = std::env::var("STONEWAVE_HOME").ok()?;
    Some(Path::new(&home).join("var").join("external_data"))
}

pub trait PathTrav {
    /// Compare two paths to check if there are path traversal.
    fn is_path_trav(&self, rel: &Path) -> Result<bool, ErrorKind>;
}

impl PathTrav for std::path::Path {
    fn is_path_trav(&self, rel: &Path) -> Result<bool, ErrorKind> {
        let base_abs = match self.canonicalize() {
            Err(err) => return Err(err.kind()),
            Ok(data) => data,
        };

        let base_abs = match base_abs.to_str() {
            None => return Err(ErrorKind::InvalidData),
            Some(da) => da,
        };

        let rel_abs = match rel.canonicalize() {
            Err(err) => return Err(err.kind()),
            Ok(data) => data,
        };

        let rel_abs = match rel_abs.to_str() {
            None => return Err(ErrorKind::InvalidData),
            Some(da) => da,
        };

        let trimmed_rel_abs: String = rel_abs.chars().take(base_abs.len()).collect();

        Ok(!trimmed_rel_abs.eq(base_abs))
    }
}
