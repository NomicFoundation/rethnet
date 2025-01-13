#[cfg(test)]
pub(super) mod tests {
    macro_rules! test_fake_sign_properties {
        () => {
            #[test]
            fn hash_with_fake_signature_same_sender() {
                use $crate::transaction::ExecutableTransaction as _;

                let transaction_request = dummy_request();

                let sender = Address::from(revm_primitives::ruint::aliases::U160::from(1));

                let signed_transaction_one = transaction_request.clone().fake_sign(sender);
                let signed_transaction_two = transaction_request.fake_sign(sender);

                let hash_one = signed_transaction_one.transaction_hash();
                let hash_two = signed_transaction_two.transaction_hash();

                assert_eq!(hash_one, hash_two);
            }

            #[test]
            fn hash_with_fake_signature_different_senders() {
                use $crate::transaction::ExecutableTransaction as _;

                let transaction_request = dummy_request();

                let sender_one = Address::from(revm_primitives::ruint::aliases::U160::from(1));
                let sender_two = Address::from(revm_primitives::ruint::aliases::U160::from(2));

                let signed_transaction_one = transaction_request.clone().fake_sign(sender_one);
                let signed_transaction_two = transaction_request.fake_sign(sender_two);

                let hash_one = signed_transaction_one.transaction_hash();
                let hash_two = signed_transaction_two.transaction_hash();

                assert_ne!(hash_one, hash_two);
            }

            #[test]
            fn recovers_fake_sender() {
                use $crate::transaction::Transaction as _;

                let transaction_request = dummy_request();

                // Fails to recover with signature error if tried to ecrocver a fake signature
                let sender: Address = "0x67091a7dd65bf4f1e95af0a479fbc782b61c129a"
                    .parse()
                    .expect("valid address");

                let signed_transaction = transaction_request.fake_sign(sender);
                assert_eq!(*signed_transaction.caller(), sender);
            }
        };
    }

    //  Needs to be `pub(crate`), otherwise export doesn't work.
    pub(crate) use test_fake_sign_properties;
}
