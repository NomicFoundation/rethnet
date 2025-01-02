// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./test.sol";
import "./Vm.sol";

// Test that the contract environment related config values are passed on correctly
contract ContractEnvironmentTest is DSTest {
    Vm constant vm = Vm(HEVM_ADDRESS);
    
    function chainId() internal view returns (uint256 id) {
        assembly {
            id := chainid()
        }
    }

    function testEnvironment() public {
        assertEq(msg.sender, 0x976EA74026E726554dB657fA54763abd0C3a0aa9, "sender account is incorrect");
        assertEq(chainId(), 12, "chain id is incorrect");
        assertEq(block.number, 23, "block number is incorrect");
        assertEq(block.timestamp, 45, "timestamp is incorrect");
    }
    
    function testContextIsTest() public {
        assertEq(vm.isContext(Vm.ExecutionContext.Test), true);
    }
}
