//! Implementations of [`Environment`](crate::Group::Environment) cheatcodes.

use std::env;

use alloy_dyn_abi::DynSolType;
use alloy_sol_types::SolValue;

use crate::{
    config::ExecutionContextConfig,
    string, Cheatcode, Cheatcodes, Error, Result,
    Vm::{
        envAddress_0Call, envAddress_1Call, envBool_0Call, envBool_1Call, envBytes32_0Call,
        envBytes32_1Call, envBytes_0Call, envBytes_1Call, envExistsCall, envInt_0Call,
        envInt_1Call, envOr_0Call, envOr_10Call, envOr_11Call, envOr_12Call, envOr_13Call,
        envOr_1Call, envOr_2Call, envOr_3Call, envOr_4Call, envOr_5Call, envOr_6Call, envOr_7Call,
        envOr_8Call, envOr_9Call, envString_0Call, envString_1Call, envUint_0Call, envUint_1Call,
        isContextCall, setEnvCall, ExecutionContext,
    },
};

impl Cheatcode for setEnvCall {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name: key, value } = self;
        if key.is_empty() {
            Err(fmt_err!("environment variable key can't be empty"))
        } else if key.contains('=') {
            Err(fmt_err!(
                "environment variable key can't contain equal sign `=`"
            ))
        } else if key.contains('\0') {
            Err(fmt_err!(
                "environment variable key can't contain NUL character `\\0`"
            ))
        } else if value.contains('\0') {
            Err(fmt_err!(
                "environment variable value can't contain NUL character `\\0`"
            ))
        } else {
            env::set_var(key, value);
            Ok(Vec::default())
        }
    }
}

impl Cheatcode for envExistsCall {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name } = self;
        Ok(env::var(name).is_ok().abi_encode())
    }
}

impl Cheatcode for envBool_0Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name } = self;
        env(name, &DynSolType::Bool)
    }
}

impl Cheatcode for envUint_0Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name } = self;
        env(name, &DynSolType::Uint(256))
    }
}

impl Cheatcode for envInt_0Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name } = self;
        env(name, &DynSolType::Int(256))
    }
}

impl Cheatcode for envAddress_0Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name } = self;
        env(name, &DynSolType::Address)
    }
}

impl Cheatcode for envBytes32_0Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name } = self;
        env(name, &DynSolType::FixedBytes(32))
    }
}

impl Cheatcode for envString_0Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name } = self;
        env(name, &DynSolType::String)
    }
}

impl Cheatcode for envBytes_0Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name } = self;
        env(name, &DynSolType::Bytes)
    }
}

impl Cheatcode for envBool_1Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, delim } = self;
        env_array(name, delim, &DynSolType::Bool)
    }
}

impl Cheatcode for envUint_1Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, delim } = self;
        env_array(name, delim, &DynSolType::Uint(256))
    }
}

impl Cheatcode for envInt_1Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, delim } = self;
        env_array(name, delim, &DynSolType::Int(256))
    }
}

impl Cheatcode for envAddress_1Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, delim } = self;
        env_array(name, delim, &DynSolType::Address)
    }
}

impl Cheatcode for envBytes32_1Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, delim } = self;
        env_array(name, delim, &DynSolType::FixedBytes(32))
    }
}

impl Cheatcode for envString_1Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, delim } = self;
        env_array(name, delim, &DynSolType::String)
    }
}

impl Cheatcode for envBytes_1Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, delim } = self;
        env_array(name, delim, &DynSolType::Bytes)
    }
}

// bool
impl Cheatcode for envOr_0Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, defaultValue } = self;
        env_default(name, defaultValue, &DynSolType::Bool)
    }
}

// uint256
impl Cheatcode for envOr_1Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, defaultValue } = self;
        env_default(name, defaultValue, &DynSolType::Uint(256))
    }
}

// int256
impl Cheatcode for envOr_2Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, defaultValue } = self;
        env_default(name, defaultValue, &DynSolType::Int(256))
    }
}

// address
impl Cheatcode for envOr_3Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, defaultValue } = self;
        env_default(name, defaultValue, &DynSolType::Address)
    }
}

// bytes32
impl Cheatcode for envOr_4Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, defaultValue } = self;
        env_default(name, defaultValue, &DynSolType::FixedBytes(32))
    }
}

// string
impl Cheatcode for envOr_5Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, defaultValue } = self;
        env_default(name, defaultValue, &DynSolType::String)
    }
}

// bytes
impl Cheatcode for envOr_6Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { name, defaultValue } = self;
        env_default(name, defaultValue, &DynSolType::Bytes)
    }
}

// bool[]
impl Cheatcode for envOr_7Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self {
            name,
            delim,
            defaultValue,
        } = self;
        env_array_default(name, delim, defaultValue, &DynSolType::Bool)
    }
}

// uint256[]
impl Cheatcode for envOr_8Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self {
            name,
            delim,
            defaultValue,
        } = self;
        env_array_default(name, delim, defaultValue, &DynSolType::Uint(256))
    }
}

// int256[]
impl Cheatcode for envOr_9Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self {
            name,
            delim,
            defaultValue,
        } = self;
        env_array_default(name, delim, defaultValue, &DynSolType::Int(256))
    }
}

// address[]
impl Cheatcode for envOr_10Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self {
            name,
            delim,
            defaultValue,
        } = self;
        env_array_default(name, delim, defaultValue, &DynSolType::Address)
    }
}

// bytes32[]
impl Cheatcode for envOr_11Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self {
            name,
            delim,
            defaultValue,
        } = self;
        env_array_default(name, delim, defaultValue, &DynSolType::FixedBytes(32))
    }
}

// string[]
impl Cheatcode for envOr_12Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self {
            name,
            delim,
            defaultValue,
        } = self;
        env_array_default(name, delim, defaultValue, &DynSolType::String)
    }
}

// bytes[]
impl Cheatcode for envOr_13Call {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self {
            name,
            delim,
            defaultValue,
        } = self;
        let default = defaultValue.clone();
        env_array_default(name, delim, &default, &DynSolType::Bytes)
    }
}

impl Cheatcode for isContextCall {
    fn apply(&self, state: &mut Cheatcodes) -> Result {
        let Self {
            context: context_arg,
        } = self;
        let configured_context = &state.config.execution_context;

        let group_match = matches!(
            (configured_context, context_arg),
            (
                &ExecutionContextConfig::Test
                    | &ExecutionContextConfig::Snapshot
                    | &ExecutionContextConfig::Coverage,
                ExecutionContext::TestGroup,
            )
        );

        let exact_match = matches!(
            (configured_context, context_arg),
            (ExecutionContextConfig::Coverage, ExecutionContext::Coverage)
                | (ExecutionContextConfig::Snapshot, ExecutionContext::Snapshot)
                | (ExecutionContextConfig::Test, ExecutionContext::Test)
                | (ExecutionContextConfig::Unknown, ExecutionContext::Unknown)
        );

        Ok((group_match || exact_match).abi_encode())
    }
}

fn env(key: &str, ty: &DynSolType) -> Result {
    get_env(key).and_then(|val| string::parse(&val, ty).map_err(map_env_err(key, &val)))
}

fn env_default<T: SolValue>(key: &str, default: &T, ty: &DynSolType) -> Result {
    Ok(env(key, ty).unwrap_or_else(|_err| default.abi_encode()))
}

fn env_array(key: &str, delim: &str, ty: &DynSolType) -> Result {
    get_env(key).and_then(|val| {
        string::parse_array(val.split(delim).map(str::trim), ty).map_err(map_env_err(key, &val))
    })
}

fn env_array_default<T: SolValue>(key: &str, delim: &str, default: &T, ty: &DynSolType) -> Result {
    Ok(env_array(key, delim, ty).unwrap_or_else(|_err| default.abi_encode()))
}

fn get_env(key: &str) -> Result<String> {
    match env::var(key) {
        Ok(val) => Ok(val),
        Err(env::VarError::NotPresent) => Err(fmt_err!("environment variable {key:?} not found")),
        Err(env::VarError::NotUnicode(s)) => Err(fmt_err!(
            "environment variable {key:?} was not valid unicode: {s:?}"
        )),
    }
}

/// Converts the error message of a failed parsing attempt to a more
/// user-friendly message that doesn't leak the value.
fn map_env_err<'a>(key: &'a str, value: &'a str) -> impl FnOnce(Error) -> Error + 'a {
    move |e| {
        // failed parsing <value> as type `uint256`: parser error:
        // <value>
        //   ^
        //   expected at least one digit
        let mut e = e.to_string();
        e = e.replacen(&format!("\"{value}\""), &format!("${key}"), 1);
        e = e.replacen(&format!("\n{value}\n"), &format!("\n${key}\n"), 1);
        Error::from(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_env_uint() {
        let key = "parse_env_uint";
        let value = "t";
        env::set_var(key, value);

        let err = env(key, &DynSolType::Uint(256)).unwrap_err().to_string();
        assert_eq!(err.matches("$parse_env_uint").count(), 2, "{err:?}");
        env::remove_var(key);
    }
}
