// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity 0.8.18;

import "ds-test/test.sol";
import "cheats/Vm.sol";

contract ProjectRootTest is DSTest {
    Vm constant vm = Vm(HEVM_ADDRESS);
    bytes public manifestDirBytes;

    function testProjectRoot() public {
        manifestDirBytes = bytes(vm.envString("CARGO_MANIFEST_DIR"));
        bytes memory expectedRootDir = abi.encodePacked(manifestDirBytes, "/tests/testdata");
        assertEq(vm.projectRoot(), string(expectedRootDir));
    }
}
