use anyhow::anyhow;
use arrow::array::RecordBatch;

use crate::TableFunction;

#[derive(Debug)]
pub struct ErrorProducer {}

impl TableFunction for ErrorProducer {
    fn process(&mut self, _input: RecordBatch) -> anyhow::Result<Option<RecordBatch>> {
        Err(anyhow!("Error Process"))
    }

    fn finalize(&mut self) -> anyhow::Result<Option<RecordBatch>> {
        Err(anyhow!("Error Finalize"))
    }
}
