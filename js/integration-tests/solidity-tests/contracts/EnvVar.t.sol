// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./test.sol";
import "./Vm.sol";

contract EnvVarTest is DSTest {
    Vm constant vm = Vm(HEVM_ADDRESS);

    function testGetEnv() public {
        string memory key = "_EDR_SOLIDITY_TESTS_GET_ENV_TEST_KEY";
        string memory val = "_edrSolidityTestsGetEnvTestVal";
        string memory result = vm.envString(key);
        assertEq(result, val);
    }
}
