use edr_solidity::{
    artifacts::{CompilerInput, CompilerOutput},
    compiler::create_models_and_decode_bytecodes,
};

use crate::{data::ProviderData, time::TimeSinceEpoch, ProviderError, ProviderSpec};

pub fn handle_add_compilation_result<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    solc_version: String,
    compiler_input: CompilerInput,
    compiler_output: CompilerOutput,
) -> Result<bool, ProviderError<ChainSpecT>> {
    if let Err(error) = add_compilation_result_inner::<ChainSpecT, TimerT>(
        data,
        solc_version,
        compiler_input,
        compiler_output,
    ) {
        data.logger_mut()
            .print_contract_decoding_error(&error.to_string())
            .map_err(ProviderError::Logger)?;
        Ok(false)
    } else {
        Ok(true)
    }
}

fn add_compilation_result_inner<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    solc_version: String,
    compiler_input: CompilerInput,
    compiler_output: CompilerOutput,
) -> Result<(), ProviderError<ChainSpecT>> {
    let contracts =
        create_models_and_decode_bytecodes(solc_version, &compiler_input, &compiler_output)
            .map_err(|err| ProviderError::SolcDecoding(err.to_string()))?;

    let contract_decoder = data.contract_decoder();
    for contract in contracts {
        contract_decoder.add_contract_metadata(contract);
    }

    Ok(())
}
