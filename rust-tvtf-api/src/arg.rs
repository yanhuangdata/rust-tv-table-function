use serde::Deserialize;

pub type Args = Vec<Arg>;

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", content = "value")]
#[serde(rename_all = "lowercase")]
pub enum Arg {
    Int(i64),
    String(String),
    Bool(bool),
    Float(f64),
    Timestamp(i64),
    Interval(String),
    Column(String),
}

impl Arg {
    pub fn is_scalar(&self) -> bool {
        use Arg as T;
        matches!(
            self,
            T::Int(_) | T::String(_) | T::Bool(_) | T::Float(_) | T::Timestamp(_) | T::Interval(_)
        )
    }

    pub fn is_column(&self) -> bool {
        use Arg as T;
        matches!(self, T::Column(_))
    }
}

#[cfg(test)]
mod tests {

    use anyhow::Context;
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_args() -> anyhow::Result<()> {
        serde_json::from_value::<Args>(
                    json! {[{"type":"column","value":"test_cte"},{"type":"string","value":"output_test_name"}]},
                ).context("Failed to parse parameters")?;
        Ok(())
    }
}
