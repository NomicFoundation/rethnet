// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity 0.8.18;

import "ds-test/test.sol";
import "cheats/Vm.sol";

contract ExecutionContextTest is DSTest {
    Vm constant vm = Vm(HEVM_ADDRESS);

    function testContext() public {
        assertEq(vm.isContext(Vm.ExecutionContext.Test), true);
        assertEq(vm.isContext(Vm.ExecutionContext.TestGroup), true);
        assertEq(vm.isContext(Vm.ExecutionContext.Coverage), false);
        assertEq(vm.isContext(Vm.ExecutionContext.Snapshot), false);
        assertEq(vm.isContext(Vm.ExecutionContext.Unknown), false);
    }
}
