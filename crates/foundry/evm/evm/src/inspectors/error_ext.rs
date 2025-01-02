use alloy_primitives::Bytes;
use alloy_sol_types::SolError;

/// An extension trait for `std::error::Error` for ABI encoding.
pub(crate) trait ErrorExt: std::error::Error {
    /// ABI-encodes the error using `Revert(string)`.
    fn abi_encode_revert(&self) -> Bytes;
}

impl<T: std::error::Error> ErrorExt for T {
    fn abi_encode_revert(&self) -> Bytes {
        alloy_sol_types::Revert::from(self.to_string())
            .abi_encode()
            .into()
    }
}
