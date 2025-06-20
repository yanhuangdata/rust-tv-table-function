use crate::funcs::*;
use anyhow::Context;
use rust_tvtf_api::Signature;
use rust_tvtf_api::{FunctionRegistry, TableFunction, arg::ArgType};
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
                OutputCsv::new(ctx.arguments).map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .signature(vec![ArgType::String])
            .signature(vec![ArgType::String, ArgType::Bool])
            .build()
            .context("create `output_csv` registry failed")?,
        FunctionRegistry::builder()
            .name("transaction")
            .init(Arc::new(|ctx| {
                TransFunction::new(ctx.arguments, ctx.named_arguments)
                    .map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .signature(
                Signature::builder()
                    .parameter(ArgType::String) // fields
                    .parameter((Some("starts_with"), ArgType::String, Some("")))
                    .parameter((Some("starts_with_regex"), ArgType::String, Some("")))
                    .parameter((Some("starts_if"), ArgType::String, Some("")))
                    .parameter((Some("ends_with"), ArgType::String, Some("")))
                    .parameter((Some("ends_with_regex"), ArgType::String, Some("")))
                    .parameter((Some("ends_if"), ArgType::String, Some("")))
                    .parameter((Some("max_span"), ArgType::Int, Some(1000)))
                    .build()
                    .context("Failed to build signature parameters")?,
            )
            .build()
            .context("create `output_csv` registry failed")?,
    ])
}
