// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.24;

import "./test.sol";
import "./Vm.sol";

contract TransientStorageStore {
  uint value;

  function tstore(uint key, uint val) public {
    assembly {
      tstore(key, val)
    }
  }

  function tload(uint key) public view returns (uint) {
    uint val;
    assembly {
      val := tload(key)
    }
    return val;
  }
}

contract IsolateTest is DSTest {
    TransientStorageStore transientStorageStore;

    function setUp() public {
      transientStorageStore = new TransientStorageStore();
    }

    function testIsolateTest() public {
        transientStorageStore.tstore(1, 2);

        // a normal transaction would return 2 here because the value is stored in transient storage,
        // but if isolate mode is enabled, the value should be 0
        uint256 val = transientStorageStore.tload(1);

        assertEq(val, 0, "test wasn't called with isolate mode enabld");
    }
}
