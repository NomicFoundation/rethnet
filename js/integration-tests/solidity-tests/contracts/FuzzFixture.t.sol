// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./test.sol";
import "./Vm.sol";

// Contract to be tested with overflow vulnerability
contract IdentityContract {
    function identity(uint256 amount) pure public returns(uint256) {
        require(amount != 7191815684697958081204101901807852913954269296144377099693178655035380638910, "Got value from fixture");
        return amount;
    }
}

// Test that fuzz fixtures specified in Solidity are not supported.
contract FuzzFixtureTest is DSTest {
    Vm constant vm = Vm(HEVM_ADDRESS);
    
    IdentityContract testDummy;

    uint256[] public fixtureAmount = [
        // This is a random value
        7191815684697958081204101901807852913954269296144377099693178655035380638910
    ];

    function setUp() public {
        testDummy = new IdentityContract();
    }

    function testFuzzDummy(uint256 amount) public {
        assertEq(testDummy.identity(amount), amount);
    }
}
