import { assert } from "chai";
import path from "path";

// The benchmarks assume that the EDR version used by
// Hardhat is the one in the workspace, instead of the one installed
// from npm. This test checks that this is the case.
it("uses the workspace version of EDR", async function () {
  const hardhatPath = require.resolve("hardhat");

  const edrPath = require.resolve("@ignored/edr", {
    paths: [hardhatPath],
  });

  const expectedPath = path.resolve(
    __dirname,
    "..",
    "..",
    "..",
    "..",
    "..",
    "crates",
    "edr_napi",
    "index.js"
  );

  assert.equal(edrPath, expectedPath);
});
