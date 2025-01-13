use std::sync::OnceLock;

use alloy_rlp::Encodable as _;
use revm_primitives::VERSIONED_HASH_VERSION_KZG;
use sha2::Digest;

use crate::{
    eips::{eip2930, eip4844::EnvKzgSettings, eip7702},
    transaction::{self, ExecutableTransaction, Transaction, TxKind},
    utils::enveloped,
    Address, Blob, Bytes, Bytes48, B256, U256,
};

/// An EIP-4844 pooled transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Eip4844 {
    payload: transaction::signed::Eip4844,
    blobs: Vec<Blob>,
    commitments: Vec<Bytes48>,
    proofs: Vec<Bytes48>,
    rlp_encoding: OnceLock<Bytes>,
}

#[derive(Debug, thiserror::Error)]
pub enum CreationError {
    #[error("Number of blobs ({actual}) does not match the payload's number of blob hashes ({expected}).")]
    BlobCount { expected: usize, actual: usize },
    #[error("The versioned hash of the commitment at index {idx} does not match the payload's blob hash. Expected: {expected}, actual: {actual}.")]
    InvalidCommitment {
        idx: usize,
        expected: B256,
        actual: B256,
    },
    #[error("Number of commitments ({actual}) does not match the payload's number of blob hashes ({expected}).")]
    CommitmentCount { expected: usize, actual: usize },
    #[error("An error occurred while verifying the blob KZG proof: {0}")]
    KzgProof(c_kzg::Error),
    #[error("Number of proofs ({actual}) does not match the payload's number of blob hashes ({expected}).")]
    ProofCount { expected: usize, actual: usize },
    #[error("The verification of the KZG proof failed.")]
    Unverified,
}

impl Eip4844 {
    /// The type identifier for an EIP-4844 transaction.
    pub const TYPE: u8 = transaction::signed::Eip4844::TYPE;

    /// Creates a new EIP-4844 pooled transaction, if the provided blobs,
    /// commitments, and proofs are valid.
    pub fn new(
        payload: transaction::signed::Eip4844,
        blobs: Vec<Blob>,
        commitments: Vec<Bytes48>,
        proofs: Vec<Bytes48>,
        settings: &c_kzg::KzgSettings,
    ) -> Result<Self, CreationError> {
        if payload.blob_hashes.len() != blobs.len() {
            return Err(CreationError::BlobCount {
                expected: payload.blob_hashes.len(),
                actual: blobs.len(),
            });
        }

        if payload.blob_hashes.len() != commitments.len() {
            return Err(CreationError::CommitmentCount {
                expected: payload.blob_hashes.len(),
                actual: commitments.len(),
            });
        }

        if payload.blob_hashes.len() != proofs.len() {
            return Err(CreationError::ProofCount {
                expected: payload.blob_hashes.len(),
                actual: proofs.len(),
            });
        }

        let verified = c_kzg::KzgProof::verify_blob_kzg_proof_batch(
            blobs.as_slice(),
            commitments.as_slice(),
            proofs.as_slice(),
            settings,
        )
        .map_err(CreationError::KzgProof)?;

        if !verified {
            return Err(CreationError::Unverified);
        }

        let invalid_blob_hash = payload
            .blob_hashes
            .iter()
            .zip(commitments.iter())
            .enumerate()
            .find_map(|(idx, (blob_hash, commitment))| {
                let mut commitment_hash = sha2::Sha256::digest(commitment.as_ref());
                commitment_hash[0] = VERSIONED_HASH_VERSION_KZG;

                if *blob_hash == *commitment_hash {
                    None
                } else {
                    Some((idx, *blob_hash, commitment_hash))
                }
            });

        if let Some((idx, expected, actual)) = invalid_blob_hash {
            return Err(CreationError::InvalidCommitment {
                idx,
                expected,
                actual: B256::from(actual.as_ref()),
            });
        }

        Ok(Self {
            payload,
            blobs,
            commitments,
            proofs,
            rlp_encoding: OnceLock::new(),
        })
    }

    /// Returns the blobs of the pooled transaction.
    pub fn blobs(&self) -> &[Blob] {
        &self.blobs
    }

    /// Returns a reference to the blobs of the pooled transaction.
    pub fn blobs_ref(&self) -> &Vec<Blob> {
        &self.blobs
    }

    /// Returns the commitments of the pooled transaction.
    pub fn commitments(&self) -> &[Bytes48] {
        &self.commitments
    }

    /// Returns the proofs of the pooled transaction.
    pub fn proofs(&self) -> &[Bytes48] {
        &self.proofs
    }

    /// Converts the pooled transaction into its inner components.
    pub fn into_inner(
        self,
    ) -> (
        transaction::signed::Eip4844,
        Vec<Blob>,
        Vec<Bytes48>,
        Vec<Bytes48>,
    ) {
        (self.payload, self.blobs, self.commitments, self.proofs)
    }

    /// Converts the pooled transaction into its payload.
    pub fn into_payload(self) -> transaction::signed::Eip4844 {
        self.payload
    }

    /// Returns the payload of the pooled transaction.
    pub fn payload(&self) -> &transaction::signed::Eip4844 {
        &self.payload
    }

    fn rlp_payload_length(&self) -> usize {
        use alloy_rlp::Encodable;

        let blob_payload =
            c_kzg::BYTES_PER_BLOB + alloy_rlp::length_of_length(c_kzg::BYTES_PER_BLOB);

        let commitment_payload =
            c_kzg::BYTES_PER_COMMITMENT + alloy_rlp::length_of_length(c_kzg::BYTES_PER_COMMITMENT);

        let proof_payload =
            c_kzg::BYTES_PER_PROOF + alloy_rlp::length_of_length(c_kzg::BYTES_PER_PROOF);

        let blobs_payload = self.blobs.len() * blob_payload;
        let commitments_payload = self.commitments.len() * commitment_payload;
        let proofs_payload = self.proofs.len() * proof_payload;

        self.payload.length()
            + blobs_payload
            + alloy_rlp::length_of_length(blobs_payload)
            + commitments_payload
            + alloy_rlp::length_of_length(commitments_payload)
            + proofs_payload
            + alloy_rlp::length_of_length(proofs_payload)
    }
}

impl ExecutableTransaction for Eip4844 {
    fn effective_gas_price(&self, block_base_fee: U256) -> Option<U256> {
        self.payload.effective_gas_price(block_base_fee)
    }

    fn max_fee_per_gas(&self) -> Option<&U256> {
        self.payload.max_fee_per_gas()
    }

    fn rlp_encoding(&self) -> &Bytes {
        self.rlp_encoding.get_or_init(|| {
            let mut encoded = Vec::with_capacity(1 + self.length());
            enveloped(Self::TYPE, self, &mut encoded);
            encoded.into()
        })
    }

    fn total_blob_gas(&self) -> Option<u64> {
        self.payload.total_blob_gas()
    }

    fn transaction_hash(&self) -> &B256 {
        self.payload.transaction_hash()
    }
}

impl Transaction for Eip4844 {
    fn caller(&self) -> &Address {
        self.payload.caller()
    }

    fn gas_limit(&self) -> u64 {
        self.payload.gas_limit()
    }

    fn gas_price(&self) -> &U256 {
        self.payload.gas_price()
    }

    fn kind(&self) -> TxKind {
        self.payload.kind()
    }

    fn value(&self) -> &U256 {
        self.payload.value()
    }

    fn data(&self) -> &Bytes {
        self.payload.data()
    }

    fn nonce(&self) -> u64 {
        self.payload.nonce()
    }

    fn chain_id(&self) -> Option<u64> {
        self.payload.chain_id()
    }

    fn access_list(&self) -> &[eip2930::AccessListItem] {
        self.payload.access_list()
    }

    fn max_priority_fee_per_gas(&self) -> Option<&U256> {
        self.payload.max_priority_fee_per_gas()
    }

    fn blob_hashes(&self) -> &[B256] {
        self.payload.blob_hashes()
    }

    fn max_fee_per_blob_gas(&self) -> Option<&U256> {
        self.payload.max_fee_per_blob_gas()
    }

    fn authorization_list(&self) -> Option<&eip7702::AuthorizationList> {
        self.payload.authorization_list()
    }
}

#[repr(transparent)]
struct RlpBlob<'blob>(&'blob Blob);

impl alloy_rlp::Encodable for RlpBlob<'_> {
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        self.0.as_ref().encode(out);
    }

    fn length(&self) -> usize {
        self.0.as_ref().length()
    }
}

impl<'blob> From<&'blob Blob> for RlpBlob<'blob> {
    fn from(blob: &'blob Blob) -> Self {
        Self(blob)
    }
}

#[repr(transparent)]
struct RlpBytes48<'bytes>(&'bytes Bytes48);

impl alloy_rlp::Encodable for RlpBytes48<'_> {
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        self.0.as_ref().encode(out);
    }

    fn length(&self) -> usize {
        self.0.as_ref().length()
    }
}

impl<'bytes> From<&'bytes Bytes48> for RlpBytes48<'bytes> {
    fn from(bytes: &'bytes Bytes48) -> Self {
        Self(bytes)
    }
}

impl alloy_rlp::Decodable for Eip4844 {
    fn decode(buf: &mut &[u8]) -> alloy_rlp::Result<Self> {
        let alloy_rlp::Header {
            list,
            payload_length,
        } = alloy_rlp::Header::decode(buf)?;

        if !list {
            return Err(alloy_rlp::Error::UnexpectedString);
        }

        let started_len = buf.len();
        if started_len < payload_length {
            return Err(alloy_rlp::Error::InputTooShort);
        }

        let payload = transaction::signed::Eip4844::decode(buf)?;

        let blobs = Vec::<[u8; c_kzg::BYTES_PER_BLOB]>::decode(buf)?;
        let blobs = blobs.into_iter().map(Blob::from).collect::<Vec<_>>();

        let commitments = Vec::<[u8; c_kzg::BYTES_PER_COMMITMENT]>::decode(buf)?;
        let commitments = commitments
            .into_iter()
            .map(Bytes48::from)
            .collect::<Vec<_>>();

        let proofs = Vec::<[u8; c_kzg::BYTES_PER_PROOF]>::decode(buf)?;
        let proofs = proofs.into_iter().map(Bytes48::from).collect::<Vec<_>>();

        let consumed = started_len - buf.len();
        if consumed != payload_length {
            return Err(alloy_rlp::Error::ListLengthMismatch {
                expected: payload_length,
                got: consumed,
            });
        }

        let settings = EnvKzgSettings::Default;
        Self::new(payload, blobs, commitments, proofs, settings.get()).map_err(|_error| {
            alloy_rlp::Error::Custom("Failed to RLP decode Eip4844PooledTransaction.")
        })
    }
}

impl alloy_rlp::Encodable for Eip4844 {
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        alloy_rlp::Header {
            list: true,
            payload_length: self.rlp_payload_length(),
        }
        .encode(out);

        self.payload.encode(out);
        alloy_rlp::encode_iter(self.blobs.iter().map(RlpBlob::from), out);
        alloy_rlp::encode_iter(self.commitments.iter().map(RlpBytes48::from), out);
        alloy_rlp::encode_iter(self.proofs.iter().map(RlpBytes48::from), out);
    }

    fn length(&self) -> usize {
        let payload_length = self.rlp_payload_length();
        payload_length + alloy_rlp::length_of_length(payload_length)
    }
}
