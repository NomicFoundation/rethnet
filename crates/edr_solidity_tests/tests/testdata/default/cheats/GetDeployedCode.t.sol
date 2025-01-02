// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity 0.8.18;

import "ds-test/test.sol";
import "cheats/Vm.sol";

contract TestContract {}

contract GetDeployedCodeTest is DSTest {
    Vm constant vm = Vm(HEVM_ADDRESS);

    address public constant overrideAddress = 0x0000000000000000000000000000000000000064;

    event Payload(address sender, address target, bytes data);

    function testWithPath() public {
        bytes memory fullPath = vm.getDeployedCode("cheats/GetCode/Override.sol");
    }

    // this will set the deployed bytecode of the stateless contract to the `overrideAddress` and call the function that emits an event that will be `expectEmitted`
    function testCanEtchStatelessOverride() public {
        bytes memory code = vm.getDeployedCode("cheats/GetCode/Override.sol:Override");
        vm.etch(overrideAddress, code);

        Override over = Override(overrideAddress);

        vm.expectEmit(true, false, false, true);
        emit Payload(address(this), address(0), "hello");
        over.emitPayload(address(0), "hello");
    }

    function testWithVersion() public {
        TestContract test = new TestContract();
        bytes memory code = vm.getDeployedCode("cheats/GetDeployedCode.t.sol:TestContract:0.8.18");

        assertEq(address(test).code, code);

        vm._expectCheatcodeRevert("No matching artifact found");
        vm.getDeployedCode("cheats/GetDeployedCode.t.sol:TestContract:0.8.19");
    }
}

interface Override {
    function emitPayload(address target, bytes calldata message) external payable returns (uint256);
}
