// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.24;

contract TestFailTest {
    function testFailRevert() public pure {
        require(false, "should revert here");
    }
}
