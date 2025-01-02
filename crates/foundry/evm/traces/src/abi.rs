//! ABI related helper functions.

use alloy_json_abi::{Event, Function};
use alloy_primitives::LogData;
use eyre::{Context, Result};

/// Given a function signature string, it tries to parse it as a `Function`
pub fn get_func(sig: &str) -> Result<Function> {
    Function::parse(sig).wrap_err("could not parse function signature")
}

/// Given an event signature string, it tries to parse it as a `Event`
pub fn get_event(sig: &str) -> Result<Event> {
    Event::parse(sig).wrap_err("could not parse event signature")
}

/// Given an event without indexed parameters and a rawlog, it tries to return
/// the event with the proper indexed parameters. Otherwise, it returns the
/// original event.
pub fn get_indexed_event(mut event: Event, raw_log: &LogData) -> Event {
    if !event.anonymous && raw_log.topics().len() > 1 {
        let indexed_params = raw_log.topics().len() - 1;
        let num_inputs = event.inputs.len();
        let num_address_params = event.inputs.iter().filter(|p| p.ty == "address").count();

        event
            .inputs
            .iter_mut()
            .enumerate()
            .for_each(|(index, param)| {
                if param.name.is_empty() {
                    param.name = format!("param{index}");
                }
                if num_inputs == indexed_params
                    || (num_address_params == indexed_params && param.ty == "address")
                {
                    param.indexed = true;
                }
            });
    }
    event
}

#[cfg(test)]
mod tests {
    use alloy_dyn_abi::{DynSolValue, EventExt};
    use alloy_primitives::{Address, B256, U256};

    use super::*;

    #[test]
    fn test_get_func() {
        let func = get_func("function foo(uint256 a, uint256 b) returns (uint256)");
        assert!(func.is_ok());
        let func = func.unwrap();
        assert_eq!(func.name, "foo");
        assert_eq!(func.inputs.len(), 2);
        assert_eq!(func.inputs[0].ty, "uint256");
        assert_eq!(func.inputs[1].ty, "uint256");

        // Stripped down function, which [Function] can parse.
        let func = get_func("foo(bytes4 a, uint8 b)(bytes4)");
        assert!(func.is_ok());
        let func = func.unwrap();
        assert_eq!(func.name, "foo");
        assert_eq!(func.inputs.len(), 2);
        assert_eq!(func.inputs[0].ty, "bytes4");
        assert_eq!(func.inputs[1].ty, "uint8");
        assert_eq!(func.outputs[0].ty, "bytes4");
    }

    #[test]
    fn test_indexed_only_address() {
        let event = get_event("event Ev(address,uint256,address)").unwrap();

        let param0 = B256::random();
        let param1 = vec![3; 32];
        let param2 = B256::random();
        let log = LogData::new_unchecked(
            vec![event.selector(), param0, param2],
            param1.clone().into(),
        );
        let event = get_indexed_event(event, &log);

        assert_eq!(event.inputs.len(), 3);

        // Only the address fields get indexed since total_params > num_indexed_params
        let parsed = event.decode_log(&log, false).unwrap();

        assert_eq!(event.inputs.iter().filter(|param| param.indexed).count(), 2);
        assert_eq!(
            parsed.indexed[0],
            DynSolValue::Address(Address::from_word(param0))
        );
        assert_eq!(
            parsed.body[0],
            DynSolValue::Uint(U256::from_be_bytes([3; 32]), 256)
        );
        assert_eq!(
            parsed.indexed[1],
            DynSolValue::Address(Address::from_word(param2))
        );
    }

    #[test]
    fn test_indexed_all() {
        let event = get_event("event Ev(address,uint256,address)").unwrap();

        let param0 = B256::random();
        let param1 = vec![3; 32];
        let param2 = B256::random();
        let log = LogData::new_unchecked(
            vec![event.selector(), param0, B256::from_slice(&param1), param2],
            vec![].into(),
        );
        let event = get_indexed_event(event, &log);

        assert_eq!(event.inputs.len(), 3);

        // All parameters get indexed since num_indexed_params == total_params
        assert_eq!(event.inputs.iter().filter(|param| param.indexed).count(), 3);
        let parsed = event.decode_log(&log, false).unwrap();

        assert_eq!(
            parsed.indexed[0],
            DynSolValue::Address(Address::from_word(param0))
        );
        assert_eq!(
            parsed.indexed[1],
            DynSolValue::Uint(U256::from_be_bytes([3; 32]), 256)
        );
        assert_eq!(
            parsed.indexed[2],
            DynSolValue::Address(Address::from_word(param2))
        );
    }
}
