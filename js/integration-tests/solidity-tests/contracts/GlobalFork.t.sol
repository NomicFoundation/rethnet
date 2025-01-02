// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

// Test that setting `eth_rpc_url` and `fork_block_number` works correctly
contract GlobalForkTest {
    function testBlockNumber() public view {
        require(block.number == 20_000_000, "Block number is not 20_000_000");
    }
}
