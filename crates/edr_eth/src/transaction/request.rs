mod eip155;
mod eip1559;
mod eip2930;
mod eip4844;
mod legacy;

use k256::SecretKey;

pub use self::{
    eip155::Eip155, eip1559::Eip1559, eip2930::Eip2930, eip4844::Eip4844, legacy::Legacy,
};
use super::{
    signed::{FakeSign, Sign},
    Request, Signed,
};
use crate::{signature::SignatureError, Address, U256};

impl Request {
    /// Retrieves the instance's chain ID.
    pub fn chain_id(&self) -> Option<u64> {
        match self {
            Request::Legacy(_) => None,
            Request::Eip155(transaction) => Some(transaction.chain_id),
            Request::Eip2930(transaction) => Some(transaction.chain_id),
            Request::Eip1559(transaction) => Some(transaction.chain_id),
            Request::Eip4844(transaction) => Some(transaction.chain_id),
        }
    }

    /// Retrieves the instance's gas price.
    pub fn gas_price(&self) -> &U256 {
        match self {
            Request::Legacy(transaction) => &transaction.gas_price,
            Request::Eip155(transaction) => &transaction.gas_price,
            Request::Eip2930(transaction) => &transaction.gas_price,
            Request::Eip1559(transaction) => &transaction.max_fee_per_gas,
            Request::Eip4844(transaction) => &transaction.max_fee_per_gas,
        }
    }

    /// Retrieves the instance's max fee per gas, if it exists.
    pub fn max_fee_per_gas(&self) -> Option<&U256> {
        match self {
            Request::Legacy(_) | Request::Eip155(_) | Request::Eip2930(_) => None,
            Request::Eip1559(transaction) => Some(&transaction.max_fee_per_gas),
            Request::Eip4844(transaction) => Some(&transaction.max_fee_per_gas),
        }
    }

    /// Retrieves the instance's max priority fee per gas, if it exists.
    pub fn max_priority_fee_per_gas(&self) -> Option<&U256> {
        match self {
            Request::Legacy(_) | Request::Eip155(_) | Request::Eip2930(_) => None,
            Request::Eip1559(transaction) => Some(&transaction.max_priority_fee_per_gas),
            Request::Eip4844(transaction) => Some(&transaction.max_priority_fee_per_gas),
        }
    }

    /// Retrieves the instance's nonce.
    pub fn nonce(&self) -> u64 {
        match self {
            Request::Legacy(transaction) => transaction.nonce,
            Request::Eip155(transaction) => transaction.nonce,
            Request::Eip2930(transaction) => transaction.nonce,
            Request::Eip1559(transaction) => transaction.nonce,
            Request::Eip4844(transaction) => transaction.nonce,
        }
    }

    pub fn sign(self, secret_key: &SecretKey) -> Result<Signed, SignatureError> {
        Ok(match self {
            Request::Legacy(transaction) => transaction.sign(secret_key)?.into(),
            Request::Eip155(transaction) => transaction.sign(secret_key)?.into(),
            Request::Eip2930(transaction) => transaction.sign(secret_key)?.into(),
            Request::Eip1559(transaction) => transaction.sign(secret_key)?.into(),
            Request::Eip4844(transaction) => transaction.sign(secret_key)?.into(),
        })
    }
}

impl FakeSign for Request {
    type Signed = Signed;

    fn fake_sign(self, sender: Address) -> Signed {
        match self {
            Request::Legacy(transaction) => transaction.fake_sign(sender).into(),
            Request::Eip155(transaction) => transaction.fake_sign(sender).into(),
            Request::Eip2930(transaction) => transaction.fake_sign(sender).into(),
            Request::Eip1559(transaction) => transaction.fake_sign(sender).into(),
            Request::Eip4844(transaction) => transaction.fake_sign(sender).into(),
        }
    }
}

impl Sign for Request {
    type Signed = Signed;

    unsafe fn sign_for_sender_unchecked(
        self,
        secret_key: &SecretKey,
        caller: Address,
    ) -> Result<Signed, SignatureError> {
        Ok(match self {
            Request::Legacy(transaction) => transaction
                .sign_for_sender_unchecked(secret_key, caller)?
                .into(),
            Request::Eip155(transaction) => transaction
                .sign_for_sender_unchecked(secret_key, caller)?
                .into(),
            Request::Eip2930(transaction) => transaction
                .sign_for_sender_unchecked(secret_key, caller)?
                .into(),
            Request::Eip1559(transaction) => transaction
                .sign_for_sender_unchecked(secret_key, caller)?
                .into(),
            Request::Eip4844(transaction) => transaction
                .sign_for_sender_unchecked(secret_key, caller)?
                .into(),
        })
    }
}

/// A transaction request and the sender's address.
#[derive(Clone, Debug)]
pub struct TransactionRequestAndSender<RequestT> {
    /// The transaction request.
    pub request: RequestT,
    /// The sender's address.
    pub sender: Address,
}
