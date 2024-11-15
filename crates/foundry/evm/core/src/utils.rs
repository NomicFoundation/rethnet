use std::{cell::RefCell, rc::Rc, sync::Arc};

use alloy_chains::NamedChain;
use alloy_json_abi::{Function, JsonAbi};
use alloy_primitives::{Address, FixedBytes, U256};
use alloy_provider::{
    network::{BlockResponse, HeaderResponse},
    Network,
};
use alloy_rpc_types::{Block, Transaction, TransactionRequest};
use eyre::ContextCompat;
use revm::{
    database_interface::WrapDatabaseRef,
    handler::register::EvmHandler,
    interpreter::{
        return_ok, CallInputs, CallOutcome, CallScheme, CallValue, CreateInputs, CreateOutcome,
        Gas, InstructionResult, InterpreterResult,
    },
    primitives::{TxKind, KECCAK_EMPTY},
    specification::hardfork::SpecId,
    wiring::{default::CreateScheme, result::EVMError},
    Database, DatabaseRef, Evm, EvmWiring, FrameOrResult, FrameResult,
};
use revm_inspector::{inspector_handle_register, Inspector};
pub use revm_state::EvmState as StateChangeset;
use revm_wiring::{result::InvalidTransaction, EvmWiring as EvmWiringTypes, TransactionValidation};

pub use crate::ic::*;
use crate::{constants::DEFAULT_CREATE2_DEPLOYER, wiring::EnvWiring, InspectorExt};

/// Depending on the configured chain id and block number this should apply any
/// specific changes
///
/// - checks for prevrandao mixhash after merge
/// - applies chain specifics: on Arbitrum `block.number` is the L1 block
/// Should be called with proper chain id (retrieved from provider if not
/// provided).
pub fn apply_chain_and_block_specific_env_changes<N: Network>(
    env: &mut EnvWiring,
    block: &N::BlockResponse,
) {
    if let Ok(chain) = NamedChain::try_from(env.cfg.chain_id) {
        let block_number = block.header().number();

        match chain {
            NamedChain::Mainnet => {
                // after merge difficulty is supplanted with prevrandao EIP-4399
                if block_number >= 15_537_351u64 {
                    env.block.difficulty = env.block.prevrandao.unwrap_or_default().into();
                }

                return;
            }
            NamedChain::Arbitrum
            | NamedChain::ArbitrumGoerli
            | NamedChain::ArbitrumNova
            | NamedChain::ArbitrumTestnet => {
                // on arbitrum `block.number` is the L1 block which is included in the
                // `l1BlockNumber` field
                if let Some(l1_block_number) = block
                    .other_fields()
                    .and_then(|other| other.get("l1BlockNumber").cloned())
                    .and_then(|l1_block_number| {
                        serde_json::from_value::<U256>(l1_block_number).ok()
                    })
                {
                    env.block.number = l1_block_number;
                }
            }
            _ => {}
        }
    }

    // if difficulty is `0` we assume it's past merge
    if block.header().difficulty().is_zero() {
        env.block.difficulty = env.block.prevrandao.unwrap_or_default().into();
    }
}

/// Given an ABI and selector, it tries to find the respective function.
pub fn get_function(
    contract_name: &str,
    selector: &FixedBytes<4>,
    abi: &JsonAbi,
) -> eyre::Result<Function> {
    abi.functions()
        .find(|func| func.selector().as_slice() == selector.as_slice())
        .cloned()
        .wrap_err(format!(
            "{contract_name} does not have the selector {selector:?}"
        ))
}

/// Configures the env for the given RPC transaction.
pub fn configure_tx_env(env: &mut EnvWiring, tx: &Transaction) {
    configure_tx_req_env(env, &tx.clone().into()).expect("cannot fail");
}

/// Configures the env for the given RPC transaction request.
pub fn configure_tx_req_env(env: &mut EnvWiring, tx: &TransactionRequest) -> eyre::Result<()> {
    let TransactionRequest {
        nonce,
        from,
        to,
        value,
        gas_price,
        gas,
        max_fee_per_gas,
        max_priority_fee_per_gas,
        max_fee_per_blob_gas,
        ref input,
        chain_id,
        ref blob_versioned_hashes,
        ref access_list,
        transaction_type: _,
        ref authorization_list,
        sidecar: _,
    } = *tx;

    // If no `to` field then set create kind: https://eips.ethereum.org/EIPS/eip-2470#deployment-transaction
    env.tx.transact_to = to.unwrap_or(TxKind::Create);
    env.tx.caller = from.ok_or_else(|| eyre::eyre!("missing `from` field"))?;
    env.tx.gas_limit = gas
        .ok_or_else(|| eyre::eyre!("missing `gas` field"))?
        .try_into()
        .map_err(|_err| eyre::eyre!("gas too large"))?;
    env.tx.nonce = nonce.ok_or_else(|| eyre::eyre!("missing `nonce` field"))?;
    env.tx.value = value.unwrap_or_default();
    env.tx.data = input.input().cloned().unwrap_or_default();
    env.tx.chain_id = chain_id;

    // Type 1, EIP-2930
    env.tx.access_list = access_list
        .clone()
        .unwrap_or_default()
        .0
        .into_iter()
        .collect();

    // Type 2, EIP-1559
    env.tx.gas_price = U256::from(gas_price.or(max_fee_per_gas).unwrap_or_default());
    env.tx.gas_priority_fee = max_priority_fee_per_gas.map(U256::from);

    // Type 3, EIP-4844
    env.tx.blob_hashes = blob_versioned_hashes.clone().unwrap_or_default();
    env.tx.max_fee_per_blob_gas = max_fee_per_blob_gas.map(U256::from);

    // Type 4, EIP-7702
    if let Some(authorization_list) = authorization_list {
        env.tx.authorization_list = Some(revm::specification::eip7702::AuthorizationList::Signed(
            authorization_list.clone(),
        ));
    }

    Ok(())
}

/// Get the gas used, accounting for refunds
pub fn gas_used(spec: SpecId, spent: u64, refunded: u64) -> u64 {
    let refund_quotient = if SpecId::enabled(spec, SpecId::LONDON) {
        5
    } else {
        2
    };
    spent - (refunded).min(spent / refund_quotient)
}

fn get_create2_factory_call_inputs(salt: U256, inputs: CreateInputs) -> CallInputs {
    let calldata = [&salt.to_be_bytes::<32>()[..], &inputs.init_code[..]].concat();
    CallInputs {
        caller: inputs.caller,
        bytecode_address: DEFAULT_CREATE2_DEPLOYER,
        target_address: DEFAULT_CREATE2_DEPLOYER,
        scheme: CallScheme::Call,
        value: CallValue::Transfer(inputs.value),
        input: calldata.into(),
        gas_limit: inputs.gas_limit,
        is_static: false,
        return_memory_offset: 0..0,
        is_eof: false,
    }
}

/// Used for routing certain CREATE2 invocations through
/// [`DEFAULT_CREATE2_DEPLOYER`].
///
/// Overrides create hook with CALL frame if
/// [`InspectorExt::should_use_create2_factory`] returns true. Keeps track of
/// overriden frames and handles outcome in the overriden `insert_call_outcome`
/// hook by inserting decoded address directly into interpreter.
///
/// Should be installed after [`revm::inspector_handle_register`] and before any
/// other registers.
pub fn create2_handler_register<EvmWiringT: EvmWiring>(handler: &mut EvmHandler<'_, EvmWiringT>)
where
    EvmWiringT::ExternalContext: InspectorExt<EvmWiringT>,
    // EVMError<<<EvmWiringT as revm_wiring::EvmWiring>::Database as Database>::Error, <<EvmWiringT
    // as revm_wiring::EvmWiring>::Transaction as TransactionValidation>::ValidationError>:
    // From<<<EvmWiringT as revm_wiring::EvmWiring>::Database as revm::Database>::Error>
{
    let create2_overrides = Rc::<RefCell<Vec<_>>>::new(RefCell::new(Vec::new()));

    let create2_overrides_inner = create2_overrides.clone();
    let old_handle = handler.execution.create.clone();
    handler.execution.create = Arc::new(
        move |ctx, mut inputs| -> Result<FrameOrResult,
            EVMError<
            <<EvmWiringT as EvmWiringTypes>::Database as Database>::Error,
            <<EvmWiringT as EvmWiringTypes>::Transaction as TransactionValidation>::ValidationError,>>
         {
            let CreateScheme::Create2 { salt } = inputs.scheme else {
                return old_handle(ctx, inputs);
            };
            if !ctx
                .external
                .should_use_create2_factory(&mut ctx.evm, &mut inputs)
            {
                return old_handle(ctx, inputs);
            }

            let gas_limit = inputs.gas_limit;

            // Generate call inputs for CREATE2 factory.
            let mut call_inputs = get_create2_factory_call_inputs(salt, *inputs);

            // Call inspector to change input or return outcome.
            let outcome = ctx.external.call(&mut ctx.evm, &mut call_inputs);

            // Push data about current override to the stack.
            create2_overrides_inner
                .borrow_mut()
                .push((ctx.evm.journaled_state.depth(), call_inputs.clone()));

            // Handle potential inspector override.
            if let Some(outcome) = outcome {
                return Ok(FrameOrResult::Result(FrameResult::Call(outcome)));
            }

            // Sanity check that CREATE2 deployer exists.
            let code_hash = ctx
                .evm
                .load_account(DEFAULT_CREATE2_DEPLOYER).map_err(EVMError::Database)?
                .data
                .info
                .code_hash;
            if code_hash == KECCAK_EMPTY {
                return Ok(FrameOrResult::Result(FrameResult::Call(CallOutcome {
                    result: InterpreterResult {
                        result: InstructionResult::Revert,
                        output: "missing CREATE2 deployer".into(),
                        gas: Gas::new(gas_limit),
                    },
                    memory_offset: 0..0,
                })));
            }

            // Create CALL frame for CREATE2 factory invocation.
            let mut frame_or_result = ctx.evm.make_call_frame(&call_inputs);

            if let Ok(FrameOrResult::Frame(frame)) = &mut frame_or_result {
                ctx.external
                    .initialize_interp(&mut frame.frame_data_mut().interpreter, &mut ctx.evm);
            }
            frame_or_result
        },
    );

    let create2_overrides_inner = create2_overrides.clone();
    let old_handle = handler.execution.insert_call_outcome.clone();

    handler.execution.insert_call_outcome =
        Arc::new(move |ctx, frame, shared_memory, mut outcome| {
            // If we are on the depth of the latest override, handle the outcome.
            if create2_overrides_inner
                .borrow()
                .last()
                .map_or(false, |(depth, _)| {
                    *depth == ctx.evm.journaled_state.depth()
                })
            {
                let (_, call_inputs) = create2_overrides_inner.borrow_mut().pop().unwrap();
                outcome = ctx.external.call_end(&mut ctx.evm, &call_inputs, outcome);

                // Decode address from output.
                let address = match outcome.instruction_result() {
                    return_ok!() => Address::try_from(outcome.output().as_ref())
                        .map_err(|_err| {
                            outcome.result = InterpreterResult {
                                result: InstructionResult::Revert,
                                output: "invalid CREATE2 factory output".into(),
                                gas: Gas::new(call_inputs.gas_limit),
                            };
                        })
                        .ok(),
                    _ => None,
                };
                frame
                    .frame_data_mut()
                    .interpreter
                    .insert_create_outcome(CreateOutcome {
                        address,
                        result: outcome.result,
                    });

                Ok(())
            } else {
                old_handle(ctx, frame, shared_memory, outcome)
            }
        });
}

/// Creates a new EVM with the given inspector.
pub fn new_evm_with_inspector<'a, EvmWiringT>(
    db: EvmWiringT::Database,
    env: Box<
        revm::wiring::default::Env<
            <EvmWiringT as EvmWiringTypes>::Block,
            <EvmWiringT as EvmWiringTypes>::Transaction,
        >,
    >,
    spec_id: EvmWiringT::Hardfork,
    inspector: EvmWiringT::ExternalContext,
) -> revm::Evm<'a, EvmWiringT>
where
    EvmWiringT: EvmWiring + 'a,
    <EvmWiringT as EvmWiringTypes>::Transaction: Default,
    <<EvmWiringT as EvmWiringTypes>::Transaction as TransactionValidation>::ValidationError:
        From<InvalidTransaction>,
    <EvmWiringT as EvmWiringTypes>::Block: Default,
    <EvmWiringT as EvmWiringTypes>::ExternalContext: Inspector<EvmWiringT>,
    <EvmWiringT as revm_wiring::EvmWiring>::ExternalContext: InspectorExt<EvmWiringT>,
    // EVMError<<<
    // EvmWiringT as revm_wiring::EvmWiring>::Database as Database>::Error, <<EvmWiringT as
    // revm_wiring::EvmWiring>::Transaction as TransactionValidation>::ValidationError>:
    // From<<<EvmWiringT as revm_wiring::EvmWiring>::Database as Database>::Error>
{
    // TODO Evm::builder can have performance issues see prior comment:
    // NOTE: We could use `revm::Evm::builder()` here, but on the current patch it
    // has some performance issues.
    revm::Evm::<'a, EvmWiringT>::builder()
        .with_db(db)
        .with_external_context(inspector)
        .with_env(env)
        .with_spec_id(spec_id)
        .append_handler_register(inspector_handle_register)
        .append_handler_register(create2_handler_register)
        .build()
}

#[cfg(test)]
mod tests {
    use revm::database_interface::EmptyDB;
    use revm_inspector::inspectors::NoOpInspector;
    use revm_wiring::EthereumWiring;

    use super::*;

    #[test]
    fn build_evm() {
        let mut db = EmptyDB::default();

        let env = Box::<
            revm::wiring::default::Env<
                revm::wiring::default::block::BlockEnv,
                revm::wiring::default::TxEnv,
            >,
        >::default();
        let spec = SpecId::LATEST;

        let mut inspector = NoOpInspector;

        let mut evm = new_evm_with_inspector::<EthereumWiring<&mut EmptyDB, &mut NoOpInspector>>(
            &mut db,
            env,
            spec,
            &mut inspector,
        );
        let result = evm.transact().unwrap();
        assert!(result.result.is_success());
    }
}
