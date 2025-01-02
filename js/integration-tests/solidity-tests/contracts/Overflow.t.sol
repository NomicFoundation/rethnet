// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./test.sol";
import "./Vm.sol";

// Contract to be tested with overflow vulnerability
contract MyContract {
    function addWithOverflow(uint256 a, uint256 b) public pure returns (uint256) {
        return a + b;
    }
}

// Test that the fuzzing catches overflows
contract OverflowTest is DSTest {
    Vm constant vm = Vm(HEVM_ADDRESS);

    MyContract public myContract;

    function setUp() public {
        myContract = new MyContract();
    }

    function testFuzzAddWithOverflow(uint256 a, uint256 b) public {
        uint256 result = myContract.addWithOverflow(a, b);
        assertEq(result, a + b);
    }
}
