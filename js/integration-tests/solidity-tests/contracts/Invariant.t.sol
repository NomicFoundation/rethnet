// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./test.sol";
import "./Vm.sol";

contract StochasticWrongContract {
    uint256 public a;
    uint256 public b;
    uint256 public both;

    function addToA(uint256 amount) external {
        // This is an intentional bug in the contract to trigger invariant failure.
        // If the conditional is removed, the invariant will pass.
        if (amount % 13 != 0) {
            a += amount;
        }
        both += amount;
    }

    function addToB(uint256 amount) external {
        b += amount;
        both += amount;
    }
}

// Test that the invariant testing works correctly by catching a bug in the contract.
contract FailingInvariantTest is DSTest {
    StochasticWrongContract wrongContract;

    function setUp() external {
        wrongContract = new StochasticWrongContract();
    }

    function invariant() external {
        assertEq(wrongContract.a() + wrongContract.b(), wrongContract.both());
    }
}

