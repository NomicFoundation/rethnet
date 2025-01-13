mod request;

use std::{ops::Deref, sync::OnceLock};

use edr_eth::{
    block,
    eips::eip2930,
    l1, signature,
    transaction::{
        self, ExecutableTransaction, HasAccessList, IsEip4844, IsLegacy, TransactionType, TxKind,
    },
    Address, Bytes, B256, U256,
};

pub use self::request::TransactionRequest;

/// RPC transaction
#[derive(Clone, Debug, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Transaction {
    /// hash of the transaction
    pub hash: B256,
    /// the number of transactions made by the sender prior to this one
    #[serde(with = "alloy_serde::quantity")]
    pub nonce: u64,
    /// hash of the block where this transaction was in
    pub block_hash: Option<B256>,
    /// block number where this transaction was in
    #[serde(with = "alloy_serde::quantity::opt")]
    pub block_number: Option<u64>,
    /// integer of the transactions index position in the block. null when its
    /// pending
    #[serde(with = "alloy_serde::quantity::opt")]
    pub transaction_index: Option<u64>,
    /// address of the sender
    pub from: Address,
    /// address of the receiver. null when its a contract creation transaction.
    pub to: Option<Address>,
    /// value transferred in Wei
    pub value: U256,
    /// gas price provided by the sender in Wei
    pub gas_price: U256,
    /// gas provided by the sender
    pub gas: U256,
    /// the data sent along with the transaction
    pub input: Bytes,
    /// chain ID
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "alloy_serde::quantity::opt"
    )]
    pub chain_id: Option<u64>,
    /// integer of the transaction type, 0x0 for legacy transactions, 0x1 for
    /// access list types, 0x2 for dynamic fees
    #[serde(
        rename = "type",
        default,
        skip_serializing_if = "Option::is_none",
        with = "alloy_serde::quantity::opt"
    )]
    pub transaction_type: Option<u8>,
    /// access list
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_list: Option<Vec<eip2930::AccessListItem>>,
    /// max fee per gas
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fee_per_gas: Option<U256>,
    /// max priority fee per gas
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_priority_fee_per_gas: Option<U256>,
    /// The maximum total fee per gas the sender is willing to pay for blob gas
    /// in wei (EIP-4844)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fee_per_blob_gas: Option<U256>,
    /// List of versioned blob hashes associated with the transaction's EIP-4844
    /// data blobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_versioned_hashes: Option<Vec<B256>>,
}

impl Transaction {
    pub fn new(
        transaction: &(impl ExecutableTransaction
              + edr_eth::transaction::Transaction
              + TransactionType
              + HasAccessList
              + IsEip4844
              + IsLegacy),
        header: Option<&block::Header>,
        transaction_index: Option<u64>,
        is_pending: bool,
        hardfork: l1::SpecId,
    ) -> Self {
        let base_fee = header.and_then(|header| header.base_fee_per_gas);
        let gas_price = if let Some(base_fee) = base_fee {
            transaction
                .effective_gas_price(base_fee)
                .unwrap_or_else(|| *transaction.gas_price())
        } else {
            // We are following Hardhat's behavior of returning the max fee per gas for
            // pending transactions.
            *transaction.gas_price()
        };

        let chain_id = transaction.chain_id().and_then(|chain_id| {
            // Following Hardhat in not returning `chain_id` for `PostEip155Legacy` legacy
            // transactions even though the chain id would be recoverable.
            if transaction.is_legacy() {
                None
            } else {
                Some(chain_id)
            }
        });

        let show_transaction_type = hardfork >= l1::SpecId::BERLIN;
        let is_typed_transaction = !transaction.is_legacy();
        let transaction_type = if show_transaction_type || is_typed_transaction {
            Some(transaction.transaction_type())
        } else {
            None
        };

        let (block_hash, block_number) = if is_pending {
            (None, None)
        } else {
            header.map(|header| (header.hash(), header.number)).unzip()
        };

        let transaction_index = if is_pending { None } else { transaction_index };

        let access_list = if transaction.has_access_list() {
            Some(transaction.access_list().to_vec())
        } else {
            None
        };

        let blob_versioned_hashes = if transaction.is_eip4844() {
            Some(transaction.blob_hashes().to_vec())
        } else {
            None
        };

        Self {
            hash: *transaction.transaction_hash(),
            nonce: transaction.nonce(),
            block_hash,
            block_number,
            transaction_index,
            from: *transaction.caller(),
            to: transaction.kind().to().copied(),
            value: *transaction.value(),
            gas_price,
            gas: U256::from(transaction.gas_limit()),
            input: transaction.data().clone(),
            chain_id,
            transaction_type: transaction_type.map(Into::<u8>::into),
            access_list,
            max_fee_per_gas: transaction.max_fee_per_gas().copied(),
            max_priority_fee_per_gas: transaction.max_priority_fee_per_gas().cloned(),
            max_fee_per_blob_gas: transaction.max_fee_per_blob_gas().cloned(),
            blob_versioned_hashes,
        }
    }
}

/// RPC transaction with signature.
#[derive(Clone, Debug, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TransactionWithSignature {
    /// Transaction
    #[serde(flatten)]
    transaction: Transaction,
    /// ECDSA recovery id
    #[serde(with = "alloy_serde::quantity")]
    pub v: u64,
    /// Y-parity for EIP-2930 and EIP-1559 transactions. In theory these
    /// transactions types shouldn't have a `v` field, but in practice they
    /// are returned by nodes.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "alloy_serde::quantity::opt"
    )]
    pub y_parity: Option<bool>,
    /// ECDSA signature r
    pub r: U256,
    /// ECDSA signature s
    pub s: U256,
}

impl Deref for TransactionWithSignature {
    type Target = Transaction;

    fn deref(&self) -> &Self::Target {
        &self.transaction
    }
}

impl TransactionWithSignature {
    /// Creates a new instance from an RPC transaction and signature.
    pub fn new(
        transaction: Transaction,
        r: U256,
        s: U256,
        v: u64,
        _y_parity: Option<bool>,
    ) -> Self {
        Self {
            transaction,
            v,
            // Following Hardhat in always returning `v` instead of `y_parity`.
            y_parity: None,
            r,
            s,
        }
    }

    /// Returns whether the transaction has odd Y parity.
    pub fn odd_y_parity(&self) -> bool {
        self.v == 1 || self.v == 28
    }

    /// Returns whether the transaction is a legacy transaction.
    pub fn is_legacy(&self) -> bool {
        matches!(self.transaction_type, None | Some(0)) && matches!(self.v, 27 | 28)
    }
}

impl From<TransactionWithSignature> for transaction::signed::Legacy {
    fn from(value: TransactionWithSignature) -> Self {
        Self {
            nonce: value.nonce,
            gas_price: value.gas_price,
            gas_limit: value.gas.to(),
            kind: if let Some(to) = value.to {
                TxKind::Call(to)
            } else {
                TxKind::Create
            },
            value: value.value,
            input: value.transaction.input,
            // SAFETY: The `from` field represents the caller address of the signed
            // transaction.
            signature: unsafe {
                signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithRecoveryId {
                        r: value.r,
                        s: value.s,
                        v: value.v,
                    },
                    value.transaction.from,
                )
            },
            hash: OnceLock::from(value.transaction.hash),
            rlp_encoding: OnceLock::new(),
        }
    }
}

impl From<TransactionWithSignature> for transaction::signed::Eip155 {
    fn from(value: TransactionWithSignature) -> Self {
        Self {
            nonce: value.nonce,
            gas_price: value.gas_price,
            gas_limit: value.gas.to(),
            kind: if let Some(to) = value.to {
                TxKind::Call(to)
            } else {
                TxKind::Create
            },
            value: value.value,
            input: value.transaction.input,
            // SAFETY: The `from` field represents the caller address of the signed
            // transaction.
            signature: unsafe {
                signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithRecoveryId {
                        r: value.r,
                        s: value.s,
                        v: value.v,
                    },
                    value.transaction.from,
                )
            },
            hash: OnceLock::from(value.transaction.hash),
            rlp_encoding: OnceLock::new(),
        }
    }
}

impl TryFrom<TransactionWithSignature> for transaction::signed::Eip2930 {
    type Error = ConversionError;

    fn try_from(value: TransactionWithSignature) -> Result<Self, Self::Error> {
        let transaction = Self {
            // SAFETY: The `from` field represents the caller address of the signed
            // transaction.
            signature: unsafe {
                signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithYParity {
                        y_parity: value.odd_y_parity(),
                        r: value.r,
                        s: value.s,
                    },
                    value.from,
                )
            },
            chain_id: value.chain_id.ok_or(ConversionError::ChainId)?,
            nonce: value.nonce,
            gas_price: value.gas_price,
            gas_limit: value.gas.to(),
            kind: if let Some(to) = value.to {
                TxKind::Call(to)
            } else {
                TxKind::Create
            },
            value: value.value,
            input: value.transaction.input,
            access_list: value
                .transaction
                .access_list
                .ok_or(ConversionError::AccessList)?
                .into(),
            hash: OnceLock::from(value.transaction.hash),
            rlp_encoding: OnceLock::new(),
        };

        Ok(transaction)
    }
}

impl TryFrom<TransactionWithSignature> for transaction::signed::Eip1559 {
    type Error = ConversionError;

    fn try_from(value: TransactionWithSignature) -> Result<Self, Self::Error> {
        let transaction = Self {
            // SAFETY: The `from` field represents the caller address of the signed
            // transaction.
            signature: unsafe {
                signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithYParity {
                        y_parity: value.odd_y_parity(),
                        r: value.r,
                        s: value.s,
                    },
                    value.from,
                )
            },
            chain_id: value.chain_id.ok_or(ConversionError::ChainId)?,
            nonce: value.nonce,
            max_priority_fee_per_gas: value
                .max_priority_fee_per_gas
                .ok_or(ConversionError::MaxPriorityFeePerGas)?,
            max_fee_per_gas: value.max_fee_per_gas.ok_or(ConversionError::MaxFeePerGas)?,
            gas_limit: value.gas.to(),
            kind: if let Some(to) = value.to {
                TxKind::Call(to)
            } else {
                TxKind::Create
            },
            value: value.value,
            input: value.transaction.input,
            access_list: value
                .transaction
                .access_list
                .ok_or(ConversionError::AccessList)?
                .into(),
            hash: OnceLock::from(value.transaction.hash),
            rlp_encoding: OnceLock::new(),
        };

        Ok(transaction)
    }
}

impl TryFrom<TransactionWithSignature> for transaction::signed::Eip4844 {
    type Error = ConversionError;

    fn try_from(value: TransactionWithSignature) -> Result<Self, Self::Error> {
        let transaction = Self {
            // SAFETY: The `from` field represents the caller address of the signed
            // transaction.
            signature: unsafe {
                signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithYParity {
                        r: value.r,
                        s: value.s,
                        y_parity: value.odd_y_parity(),
                    },
                    value.from,
                )
            },
            chain_id: value.chain_id.ok_or(ConversionError::ChainId)?,
            nonce: value.nonce,
            max_priority_fee_per_gas: value
                .max_priority_fee_per_gas
                .ok_or(ConversionError::MaxPriorityFeePerGas)?,
            max_fee_per_gas: value.max_fee_per_gas.ok_or(ConversionError::MaxFeePerGas)?,
            max_fee_per_blob_gas: value
                .max_fee_per_blob_gas
                .ok_or(ConversionError::MaxFeePerBlobGas)?,
            gas_limit: value.gas.to(),
            to: value.to.ok_or(ConversionError::ReceiverAddress)?,
            value: value.value,
            input: value.transaction.input,
            access_list: value
                .transaction
                .access_list
                .ok_or(ConversionError::AccessList)?
                .into(),
            blob_hashes: value
                .transaction
                .blob_versioned_hashes
                .ok_or(ConversionError::BlobHashes)?,
            hash: OnceLock::from(value.transaction.hash),
            rlp_encoding: OnceLock::new(),
        };

        Ok(transaction)
    }
}

impl TryFrom<TransactionWithSignature> for transaction::Signed {
    type Error = ConversionError;

    fn try_from(value: TransactionWithSignature) -> Result<Self, Self::Error> {
        let transaction_type = match value
            .transaction_type
            .map_or(Ok(transaction::Type::Legacy), transaction::Type::try_from)
        {
            Ok(r#type) => r#type,
            Err(r#type) => {
                log::warn!("Unsupported transaction type: {type}. Reverting to post-EIP 155 legacy transaction");

                // As the transaction type is not 0 or `None`, this will always result in a
                // post-EIP 155 legacy transaction.
                transaction::Type::Legacy
            }
        };

        let transaction = match transaction_type {
            transaction::Type::Legacy => {
                if value.is_legacy() {
                    Self::PreEip155Legacy(value.into())
                } else {
                    Self::PostEip155Legacy(value.into())
                }
            }
            transaction::Type::Eip2930 => Self::Eip2930(value.try_into()?),
            transaction::Type::Eip1559 => Self::Eip1559(value.try_into()?),
            transaction::Type::Eip4844 => Self::Eip4844(value.try_into()?),
        };

        Ok(transaction)
    }
}

/// Error that occurs when trying to convert the JSON-RPC `Transaction` type.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    /// Missing access list
    #[error("Missing access list")]
    AccessList,
    /// EIP-4844 transaction is missing blob (versioned) hashes
    #[error("Missing blob hashes")]
    BlobHashes,
    /// Missing chain ID
    #[error("Missing chain ID")]
    ChainId,
    /// Missing max fee per gas
    #[error("Missing max fee per gas")]
    MaxFeePerGas,
    /// Missing max priority fee per gas
    #[error("Missing max priority fee per gas")]
    MaxPriorityFeePerGas,
    /// EIP-4844 transaction is missing the max fee per blob gas
    #[error("Missing max fee per blob gas")]
    MaxFeePerBlobGas,
    /// EIP-4844 transaction is missing the receiver (to) address
    #[error("Missing receiver (to) address")]
    ReceiverAddress,
}
