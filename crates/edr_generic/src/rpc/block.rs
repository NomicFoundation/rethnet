use derive_where::derive_where;
use edr_eth::{
    block::{BlobGas, Header},
    transaction::ExecutableTransaction,
    withdrawal::Withdrawal,
    Address, Bloom, Bytes, B256, B64, U256,
};
use edr_evm::{spec::RuntimeSpec, BlockAndTotalDifficulty, EthBlockData, EthRpcBlock};
use edr_rpc_eth::spec::GetBlockNumber;
use serde::{Deserialize, Serialize};

use crate::GenericChainSpec;

/// block object returned by `eth_getBlockBy*`
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Block<TransactionT> {
    /// Hash of the block
    pub hash: Option<B256>,
    /// hash of the parent block.
    pub parent_hash: B256,
    /// SHA3 of the uncles data in the block
    pub sha3_uncles: B256,
    /// the root of the final state trie of the block
    pub state_root: B256,
    /// the root of the transaction trie of the block
    pub transactions_root: B256,
    /// the root of the receipts trie of the block
    pub receipts_root: B256,
    /// the block number. None when its pending block.
    #[serde(with = "edr_eth::serde::optional_u64")]
    pub number: Option<u64>,
    /// the total used gas by all transactions in this block
    #[serde(with = "edr_eth::serde::u64")]
    pub gas_used: u64,
    /// the maximum gas allowed in this block
    #[serde(with = "edr_eth::serde::u64")]
    pub gas_limit: u64,
    /// the "extra data" field of this block
    pub extra_data: Bytes,
    /// the bloom filter for the logs of the block
    pub logs_bloom: Bloom,
    /// the unix timestamp for when the block was collated
    #[serde(with = "edr_eth::serde::u64")]
    pub timestamp: u64,
    /// integer of the difficulty for this blocket
    pub difficulty: U256,
    /// integer of the total difficulty of the chain until this block
    pub total_difficulty: Option<U256>,
    /// Array of uncle hashes
    #[serde(default)]
    pub uncles: Vec<B256>,
    /// Array of transaction objects, or 32 Bytes transaction hashes depending
    /// on the last given parameter
    #[serde(default)]
    pub transactions: Vec<TransactionT>,
    /// the length of the RLP encoding of this block in bytes
    #[serde(with = "edr_eth::serde::u64")]
    pub size: u64,
    /// Mix hash. None when it's a pending block.
    pub mix_hash: Option<B256>,
    /// hash of the generated proof-of-work. null when its pending block.
    pub nonce: Option<B64>,
    /// base fee per gas
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_fee_per_gas: Option<U256>,
    /// the address of the beneficiary to whom the mining rewards were given
    #[serde(skip_serializing_if = "Option::is_none")]
    pub miner: Option<Address>,
    /// withdrawals
    #[serde(skip_serializing_if = "Option::is_none")]
    pub withdrawals: Option<Vec<Withdrawal>>,
    /// withdrawals root
    #[serde(skip_serializing_if = "Option::is_none")]
    pub withdrawals_root: Option<B256>,
    /// The total amount of gas used by the transactions.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "edr_eth::serde::optional_u64"
    )]
    pub blob_gas_used: Option<u64>,
    /// A running total of blob gas consumed in excess of the target, prior to
    /// the block.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "edr_eth::serde::optional_u64"
    )]
    pub excess_blob_gas: Option<u64>,
    /// Root of the parent beacon block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_beacon_block_root: Option<B256>,
}

impl<T> GetBlockNumber for Block<T> {
    fn number(&self) -> Option<u64> {
        self.number
    }
}

impl<T> EthRpcBlock for Block<T> {
    fn state_root(&self) -> &B256 {
        &self.state_root
    }

    fn timestamp(&self) -> u64 {
        self.timestamp
    }

    fn total_difficulty(&self) -> Option<&U256> {
        self.total_difficulty.as_ref()
    }
}

/// Error that occurs when trying to convert the JSON-RPC `Block` type.
#[derive(thiserror::Error)]
#[derive_where(Debug; ChainSpecT::RpcTransactionConversionError)]
pub enum ConversionError<ChainSpecT: RuntimeSpec> {
    /// Missing hash
    #[error("Missing hash")]
    MissingHash,
    /// Missing miner
    #[error("Missing miner")]
    MissingMiner,
    /// Missing number
    #[error("Missing numbeer")]
    MissingNumber,
    /// Transaction conversion error
    #[error(transparent)]
    Transaction(ChainSpecT::RpcTransactionConversionError),
}

impl<TransactionT> TryFrom<Block<TransactionT>> for EthBlockData<GenericChainSpec>
where
    TransactionT: TryInto<
        crate::transaction::SignedWithFallbackToPostEip155,
        Error = crate::rpc::transaction::ConversionError,
    >,
{
    type Error = ConversionError<GenericChainSpec>;

    fn try_from(value: Block<TransactionT>) -> Result<Self, Self::Error> {
        let header = Header {
            parent_hash: value.parent_hash,
            ommers_hash: value.sha3_uncles,
            beneficiary: value.miner.ok_or(ConversionError::MissingMiner)?,
            state_root: value.state_root,
            transactions_root: value.transactions_root,
            receipts_root: value.receipts_root,
            logs_bloom: value.logs_bloom,
            difficulty: value.difficulty,
            number: value.number.ok_or(ConversionError::MissingNumber)?,
            gas_limit: value.gas_limit,
            gas_used: value.gas_used,
            timestamp: value.timestamp,
            extra_data: value.extra_data,
            // Do what Hardhat does and accept the remote blocks with missing
            // nonce or mix hash. See https://github.com/NomicFoundation/edr/issues/635
            mix_hash: value.mix_hash.unwrap_or_default(),
            nonce: value.nonce.unwrap_or_default(),
            base_fee_per_gas: value.base_fee_per_gas,
            withdrawals_root: value.withdrawals_root,
            blob_gas: value.blob_gas_used.and_then(|gas_used| {
                value.excess_blob_gas.map(|excess_gas| BlobGas {
                    gas_used,
                    excess_gas,
                })
            }),
            parent_beacon_block_root: value.parent_beacon_block_root,
        };

        let transactions = value
            .transactions
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ConversionError::Transaction)?;

        let hash = value.hash.ok_or(ConversionError::MissingHash)?;

        Ok(Self {
            header,
            transactions,
            ommer_hashes: value.uncles,
            withdrawals: value.withdrawals,
            hash,
            rlp_size: value.size,
        })
    }
}

impl<BlockT: edr_evm::Block<SignedTransactionT>, SignedTransactionT: ExecutableTransaction>
    From<BlockAndTotalDifficulty<BlockT, SignedTransactionT>> for crate::rpc::block::Block<B256>
{
    fn from(value: BlockAndTotalDifficulty<BlockT, SignedTransactionT>) -> Self {
        let transactions = value
            .block
            .transactions()
            .iter()
            .map(|tx| *tx.transaction_hash())
            .collect();

        let header = value.block.header();
        crate::rpc::block::Block {
            hash: Some(*value.block.block_hash()),
            parent_hash: header.parent_hash,
            sha3_uncles: header.ommers_hash,
            state_root: header.state_root,
            transactions_root: header.transactions_root,
            receipts_root: header.receipts_root,
            number: Some(header.number),
            gas_used: header.gas_used,
            gas_limit: header.gas_limit,
            extra_data: header.extra_data.clone(),
            logs_bloom: header.logs_bloom,
            timestamp: header.timestamp,
            difficulty: header.difficulty,
            total_difficulty: value.total_difficulty,
            uncles: value.block.ommer_hashes().to_vec(),
            transactions,
            size: value.block.rlp_size(),
            mix_hash: Some(header.mix_hash),
            nonce: Some(header.nonce),
            base_fee_per_gas: header.base_fee_per_gas,
            miner: Some(header.beneficiary),
            withdrawals: value
                .block
                .withdrawals()
                .map(<[edr_eth::withdrawal::Withdrawal]>::to_vec),
            withdrawals_root: header.withdrawals_root,
            blob_gas_used: header.blob_gas.as_ref().map(|bg| bg.gas_used),
            excess_blob_gas: header.blob_gas.as_ref().map(|bg| bg.excess_gas),
            parent_beacon_block_root: header.parent_beacon_block_root,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use edr_evm::RemoteBlock;
    use edr_rpc_client::jsonrpc;
    use edr_rpc_eth::client::EthRpcClient;

    use crate::{rpc::transaction::TransactionWithSignature, GenericChainSpec};

    #[tokio::test(flavor = "current_thread")]
    async fn test_allow_missing_nonce_or_mix_hash() {
        // Taken from https://github.com/NomicFoundation/edr/issues/536
        const DATA: &str = r#"{
          "jsonrpc": "2.0",
          "result": {
            "author": "0xa3b079e4b54d2886ccd9cddc1335743bf1b2f0ad",
            "difficulty": "0xfffffffffffffffffffffffffffffffe",
            "extraData": "0x4e65746865726d696e64",
            "gasLimit": "0x663be0",
            "gasUsed": "0x0",
            "hash": "0x12a6d4673bc9a55b5f77d897f32eb0ff15bca32d09d3dff8013e3ff684b0a84d",
            "logsBloom": "0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "miner": "0xa3b079e4b54d2886ccd9cddc1335743bf1b2f0ad",
            "number": "0x1b000b1",
            "parentHash": "0x26046ba285c7c87ff04783347a5b63cba77d884a11a50a632ea68ba2843b3064",
            "receiptsRoot": "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
            "sha3Uncles": "0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347",
            "signature": "0x012bb653a98ede38cd88f59fb09c67a8628012e3a4bdd6925a3dee547e0c4f470aa3118d03e4c81b715200aeb34b00f7c67a0a9cd687ff26c2dbc82964d4268a00",
            "size": "0x238",
            "stateRoot": "0x62cb23bc47439bc73eef6bf9fc0e54b1a40ebb3b0cbcfefd6b13cbc79a1095b6",
            "step": 343268919,
            "totalDifficulty": "0x1b000b0ffffffffffffffffffffffffe9dc2118",
            "timestamp": "0x664d5713",
            "transactions": [],
            "transactionsRoot": "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
            "uncles": []
          },
          "id": 1
        }"#;

        type BlockResponsePayload =
            <GenericChainSpec as edr_rpc_eth::RpcSpec>::RpcBlock<TransactionWithSignature>;

        let response: jsonrpc::Response<BlockResponsePayload> = serde_json::from_str(DATA).unwrap();
        let rpc_block = match response.data {
            jsonrpc::ResponseData::Error { .. } => unreachable!("Payload above is a success"),
            jsonrpc::ResponseData::Success { result } => result,
        };

        // Not using an actual client because we do not want to use a public
        // node for less common, EVM-compatible chains that are not supported
        // by reliable providers like Alchemy or Infura.
        // Instead, we use a static response here.
        let dummy_client = Arc::new(
            EthRpcClient::<GenericChainSpec>::new("http://example.com", "<dummy>".into(), None)
                .unwrap(),
        );
        let runtime = tokio::runtime::Handle::current();

        RemoteBlock::new(rpc_block, dummy_client, runtime)
            .expect("Conversion should accept a missing nonce or mix hash");
    }
}
