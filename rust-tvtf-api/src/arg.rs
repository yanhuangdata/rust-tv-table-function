use serde::{Deserialize, Serialize};

pub type Args = Vec<Arg>;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl From<&str> for Arg {
    fn from(value: &str) -> Self {
        Arg::String(value.to_string())
    }
}

impl From<String> for Arg {
    fn from(value: String) -> Self {
        Arg::String(value)
    }
}

impl From<bool> for Arg {
    fn from(value: bool) -> Self {
        Arg::Bool(value)
    }
}

impl From<f64> for Arg {
    fn from(value: f64) -> Self {
        Arg::Float(value)
    }
}

impl From<i64> for Arg {
    fn from(value: i64) -> Self {
        Arg::Int(value)
    }
}

// for deserialize only
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct NamedArg {
    pub name: String,
    #[serde(flatten)]
    pub arg: Arg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ArgType {
    Int,
    String,
    Bool,
    Float,
    Timestamp,
    Interval,
    Column,
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
                ).context("Failed to parse arguments")?;
        Ok(())
    }

    #[test]
    fn parse_named_args() -> anyhow::Result<()> {
        serde_json::from_value::<Vec<NamedArg>>(
                    json! {[{"name":"asdf","type":"column","value":"test_cte"},{"name":"foo","type":"string","value":"output_test_name"}]},
                ).context("Failed to parse arguments")?;
        Ok(())
    }
}
