// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./test.sol";
import "./Vm.sol";

// Test that the fork cheatcode works correctly
contract ForkCheatcodeTest is DSTest {
    Vm constant vm = Vm(HEVM_ADDRESS);
    uint256 fork;
    
    function setUp() public {
        fork = vm.createSelectFork("alchemyMainnet", 20_000_000);
    }
    
    function testBlockNumber() public {
        assertEq(fork, vm.activeFork());
        assertEq(block.number, 20_000_000);
    }
}
