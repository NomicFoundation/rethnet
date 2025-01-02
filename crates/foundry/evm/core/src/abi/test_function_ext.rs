//! Commonly used traits.

use alloy_json_abi::Function;

/// Extension trait for `Function`.
pub trait TestFunctionExt {
    /// Returns whether this function should be executed as invariant test.
    fn is_invariant_test(&self) -> bool;

    /// Returns whether this function should be executed as fuzz test.
    fn is_fuzz_test(&self) -> bool;

    /// Returns whether this function is a test.
    fn is_test(&self) -> bool;

    /// Returns whether this function is a test that should fail.
    fn is_test_fail(&self) -> bool;

    /// Returns whether this function is a `setUp` function.
    fn is_setup(&self) -> bool;

    /// Returns whether this function is a fixture function.
    fn is_fixture(&self) -> bool;
}

impl TestFunctionExt for Function {
    fn is_invariant_test(&self) -> bool {
        self.name.is_invariant_test()
    }

    fn is_fuzz_test(&self) -> bool {
        // test functions that have inputs are considered fuzz tests as those inputs
        // will be fuzzed
        !self.inputs.is_empty()
    }

    fn is_test(&self) -> bool {
        self.name.is_test()
    }

    fn is_test_fail(&self) -> bool {
        self.name.is_test_fail()
    }

    fn is_setup(&self) -> bool {
        self.name.is_setup()
    }

    fn is_fixture(&self) -> bool {
        self.name.is_fixture()
    }
}

impl TestFunctionExt for String {
    fn is_invariant_test(&self) -> bool {
        self.as_str().is_invariant_test()
    }

    fn is_fuzz_test(&self) -> bool {
        self.as_str().is_fuzz_test()
    }

    fn is_test(&self) -> bool {
        self.as_str().is_test()
    }

    fn is_test_fail(&self) -> bool {
        self.as_str().is_test_fail()
    }

    fn is_setup(&self) -> bool {
        self.as_str().is_setup()
    }

    fn is_fixture(&self) -> bool {
        self.as_str().is_fixture()
    }
}

impl TestFunctionExt for str {
    fn is_invariant_test(&self) -> bool {
        self.starts_with("invariant") || self.starts_with("statefulFuzz")
    }

    #[allow(clippy::unimplemented)]
    fn is_fuzz_test(&self) -> bool {
        unimplemented!("no naming convention for fuzz tests")
    }

    fn is_test(&self) -> bool {
        self.starts_with("test")
    }

    fn is_test_fail(&self) -> bool {
        self.starts_with("testFail")
    }

    fn is_setup(&self) -> bool {
        self.eq_ignore_ascii_case("setup")
    }

    fn is_fixture(&self) -> bool {
        self.starts_with("fixture")
    }
}
