use std::{sync::Arc, vec};

use anyhow::Context;
use rust_tvtf_api::{FunctionRegistry, TableFunction, arg::ArgType};

use crate::funcs::*;

pub mod funcs;
#[rustfmt::skip]
#[allow(clippy::all)]
mod zngur_generated;

pub fn get_function_registries() -> anyhow::Result<Vec<FunctionRegistry>> {
    Ok(vec![
        FunctionRegistry::builder()
            .name("addtotals")
            .init(Arc::new(|ctx| {
                AddTotals::new(ctx.parameters).map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .signature(vec![])
            .signature(vec![ArgType::Int])
            .build()
            .context("create `addtotals` registry failed")?,
        FunctionRegistry::builder()
            .name("output_csv")
            .init(Arc::new(|ctx| {
                OutputCsv::new(ctx.parameters).map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .signature(vec![ArgType::String])
            .signature(vec![ArgType::String, ArgType::Bool])
            .build()
            .context("create `output_csv` registry failed")?,
    ])
}
