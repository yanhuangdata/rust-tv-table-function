use std::{sync::Arc, vec};

use anyhow::Context;
use rust_tvtf_api::{FunctionRegistry, Signature, TableFunction, arg::ArgType};

use crate::funcs::*;

pub mod funcs;
#[rustfmt::skip]
#[allow(clippy::all)]
mod zngur_generated;

pub fn get_function_registries() -> anyhow::Result<Vec<FunctionRegistry>> {
    Ok(vec![
        FunctionRegistry::builder()
            .name("count_column")
            .init(Arc::new(|ctx| {
                CountColumn::new(ctx.arguments).map(|f| Box::new(f) as Box<dyn TableFunction>)
            }))
            .signature(Signature::empty())
            .signature(vec![ArgType::Int])
            .build()
            .context("create `count_column` registry failed")?,
        FunctionRegistry::builder()
            .name("error_producer")
            .init(Arc::new(|_ctx| Ok(Box::new(ErrorProducer {}))))
            .signature(Signature::empty())
            .build()
            .context("create `error_producer` registry failed")?,
    ])
}
