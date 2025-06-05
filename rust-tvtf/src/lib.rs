use rust_tvtf_api::{FunctionRegistry, TableFunction};

use crate::funcs::*;

pub mod funcs;
#[rustfmt::skip]
#[allow(clippy::all)]
mod zngur_generated;

pub fn get_function_registries() -> Vec<FunctionRegistry> {
    vec![
        FunctionRegistry {
            name: "addtotals",
            // TODO: simplify this, possibly a IntoFuncInit trait
            init: Box::new(|ctx| {
                AddTotals::new(ctx.parameters).map(|f| Box::new(f) as Box<dyn TableFunction>)
            }),
        },
        FunctionRegistry {
            name: "output_csv",
            init: Box::new(|ctx| {
                OutputCsv::new(ctx.parameters).map(|f| Box::new(f) as Box<dyn TableFunction>)
            }),
        },
    ]
}
