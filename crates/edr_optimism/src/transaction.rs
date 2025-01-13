mod r#type;

/// Types for transaction gossip (aka pooled transactions).
pub mod pooled;
/// Types for transaction requests.
pub mod request;
/// Types for signed transactions.
pub mod signed;

/// An Optimism pooled transaction, used to communicate between node pools.
pub enum Pooled {
    /// Legacy transaction before EIP-155
    PreEip155Legacy(pooled::Legacy),
    /// Legacy transaction after EIP-155
    PostEip155Legacy(pooled::Eip155),
    /// EIP-2930 transaction
    Eip2930(pooled::Eip2930),
    /// EIP-1559 transaction
    Eip1559(pooled::Eip1559),
    /// EIP-4844 transaction
    Eip4844(pooled::Eip4844),
    /// Optimism deposit transaction
    Deposit(pooled::Deposit),
}

/// An Optimism transaction request.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Request {
    /// A legacy transaction request
    Legacy(request::Legacy),
    /// An EIP-155 transaction request
    Eip155(request::Eip155),
    /// An EIP-2930 transaction request
    Eip2930(request::Eip2930),
    /// An EIP-1559 transaction request
    Eip1559(request::Eip1559),
    /// An EIP-4844 transaction request
    Eip4844(request::Eip4844),
}

/// An optimism signed transaction, used in blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Signed {
    /// Legacy transaction before EIP-155
    PreEip155Legacy(signed::Legacy),
    /// Legacy transaction after EIP-155
    PostEip155Legacy(signed::Eip155),
    /// EIP-2930 transaction
    Eip2930(signed::Eip2930),
    /// EIP-1559 transaction
    Eip1559(signed::Eip1559),
    /// EIP-4844 transaction
    Eip4844(signed::Eip4844),
    /// Optimism deposit transaction
    Deposit(signed::Deposit),
}

/// The type of Optimism transaction.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Type {
    /// Optimism legacy transaction
    Legacy = signed::Legacy::TYPE,
    /// Optimism EIP-2930 transaction
    Eip2930 = signed::Eip2930::TYPE,
    /// Optimism EIP-1559 transaction
    Eip1559 = signed::Eip1559::TYPE,
    /// Optimism EIP-4844 transaction
    Eip4844 = signed::Eip4844::TYPE,
    /// Optimism deposit transaction
    Deposit = signed::Deposit::TYPE,
}
