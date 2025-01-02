//! Several ABI-related utilities for executors.

pub use foundry_cheatcodes_spec::Vm;

mod console;
pub use console::{format_units_int, format_units_uint, Console};

pub mod fmt;

mod hardhat_console;
pub use hardhat_console::{
    hh_console_selector, patch_hh_console_selector, HardhatConsole,
    HARDHAT_CONSOLE_SELECTOR_PATCHES,
};

mod test_function_ext;
pub use test_function_ext::TestFunctionExt;
