import chai, { assert, expect } from "chai";
import { TestContext } from "./testContext";
import fs from "node:fs/promises";
import { existsSync } from "node:fs";
import { FuzzTestKind } from "@ignored/edr";
import { runAllSolidityTests } from "@nomicfoundation/edr-helpers";

describe("Fuzz and invariant testing", function () {
  let testContext: TestContext;

  before(async function () {
    testContext = await TestContext.setup();
  });

  it("FailingFuzz", async function () {
    const failureDir = testContext.fuzzFailuresPersistDir;

    // Remove invariant config failure directory to make sure it's created fresh.
    await fs.rm(failureDir, {
      recursive: true,
      force: true,
    });

    const result1 = await testContext.runTestsWithStats("OverflowTest");
    assert.equal(result1.failedTests, 1);
    assert.equal(result1.totalTests, 1);

    // The fuzz failure directory should not be created if we don't set the directory
    assert.isFalse(existsSync(failureDir));

    const result2 = await testContext.runTestsWithStats("OverflowTest", {
      fuzz: {
        failurePersistDir: failureDir,
      },
    });
    assert.equal(result2.failedTests, 1);
    assert.equal(result2.totalTests, 1);

    // The fuzz failure directory should now be created
    assert.isTrue(existsSync(failureDir));
  });

  it("FuzzFixture is not supported", async function () {
    const result = await testContext.runTestsWithStats("FuzzFixtureTest", {
      fuzz: {
        runs: 256,
        dictionaryWeight: 1,
        includeStorage: false,
        includePushBytes: false,
        seed: "0x7bb9ee74aaa2abe5e2ca8a116382a9f2ed70b651e70b430e1052eff52a74ffe3",
      },
    });
    assert.equal(result.failedTests, 0);
    assert.equal(result.totalTests, 1);
  });

  // One test as steps should be sequential
  it("FailingInvariant", async function () {
    const failureDir = testContext.invariantFailuresPersistDir;
    const invariantConfig = {
      runs: 256,
      depth: 15,
      // This is false by default, we just specify it here to make it obvious to the reader.
      failOnRevert: false,
    };

    // Remove invariant config failure directory to make sure it's created fresh.
    await fs.rm(failureDir, {
      recursive: true,
      force: true,
    });

    const result1 = await testContext.runTestsWithStats(
      "FailingInvariantTest",
      {
        invariant: invariantConfig,
      }
    );
    assert.equal(result1.failedTests, 1);
    assert.equal(result1.totalTests, 1);

    // The invariant failure directory should not be created if we don't set the directory
    assert.isFalse(existsSync(failureDir));

    const results2 = await runAllSolidityTests(
      testContext.artifacts,
      testContext.matchingTest("FailingInvariantTest"),
      {
        ...testContext.defaultConfig(),
        invariant: {
          ...invariantConfig,
          failurePersistDir: failureDir,
        },
      }
    );
    assert.equal(results2.length, 1);
    assert.equal(results2[0].testResults.length, 1);
    assert.equal(results2[0].testResults[0].status, "Failure");
    const fuzzTestResult2 = results2[0].testResults[0].kind as FuzzTestKind;
    // More than one run should be needed on a fresh invariant test.
    assert.isTrue(fuzzTestResult2.runs > 1n);

    // The invariant failure directory should now be created
    assert.isTrue(existsSync(failureDir));

    const results3 = await runAllSolidityTests(
      testContext.artifacts,
      testContext.matchingTest("FailingInvariantTest"),
      {
        ...testContext.defaultConfig(),
        invariant: {
          ...invariantConfig,
          failurePersistDir: failureDir,
        },
      }
    );

    // The invariant failure directory should still be there
    assert.isTrue(existsSync(failureDir));
    assert.equal(results3.length, 1);
    assert.equal(results3[0].testResults.length, 1);
    assert.equal(results3[0].testResults[0].status, "Failure");
    const fuzzTestResult3 = results3[0].testResults[0].kind as FuzzTestKind;
    // The second time only one run should be needed, because the persisted failure is used.
    assert.equal(fuzzTestResult3.runs, 1n);
  });
});
