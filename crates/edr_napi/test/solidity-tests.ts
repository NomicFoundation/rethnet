import { assert } from "chai";

import {
  ArtifactId,
  ContractData,
  Artifact,
  runSolidityTests,
  SolidityTestRunnerConfigArgs,
  SuiteResult,
} from "..";

describe("Solidity Tests", () => {
  it("executes basic tests", async function () {
    const artifacts = [
      loadContract("./artifacts/SetupConsistencyCheck.json"),
      loadContract("./artifacts/PaymentFailureTest.json"),
    ];
    // All artifacts are test suites.
    const testSuites = artifacts.map((artifact) => artifact.id);
    const config = {
      projectRoot: __dirname,
    };

    const results = await runAllSolidityTests(artifacts, testSuites, config);

    assert.equal(results.length, artifacts.length);

    for (const res of results) {
      if (res.id.name.includes("SetupConsistencyCheck")) {
        assert.equal(res.testResults.length, 2);
        assert.equal(res.testResults[0].status, "Success");
        assert.equal(res.testResults[1].status, "Success");
      } else if (res.id.name.includes("PaymentFailureTest")) {
        assert.equal(res.testResults.length, 1);
        assert.equal(res.testResults[0].status, "Failure");
      } else {
        assert.fail("Unexpected test suite name: " + res.id.name);
      }
    }
  });

  it("throws errors", async function () {
    const artifacts = [
      loadContract("./artifacts/SetupConsistencyCheck.json"),
      loadContract("./artifacts/PaymentFailureTest.json"),
    ];
    // All artifacts are test suites.
    const testSuites = artifacts.map((artifact) => artifact.id);
    const config = {
      projectRoot: __dirname,
      // Memory limit is too large
      memoryLimit: 2n ** 65n,
    };

    await assert.isRejected(
      runAllSolidityTests(artifacts, testSuites, config),
      Error
    );
  });

  it("error callback is called if contract bytecode is invalid", async function () {
    const artifacts = [
      loadContract("./artifacts/SetupConsistencyCheck.json"),
      loadContract("./artifacts/PaymentFailureTest.json"),
    ];
    // All artifacts are test suites.
    const testSuites = artifacts.map((artifact) => artifact.id);
    const config = {
      projectRoot: __dirname,
    };

    artifacts[0].contract.bytecode = "invalid bytecode";

    await assert.isRejected(
      runAllSolidityTests(artifacts, testSuites, config),
      "Invalid hex bytecode for contract"
    );
  });
});

// Load a contract built with Hardhat into a test suite
function loadContract(artifactPath: string): Artifact {
  const compiledContract = require(artifactPath);

  const id: ArtifactId = {
    name: compiledContract.contractName,
    solcVersion: "0.8.18",
    source: compiledContract.sourceName,
  };

  const contract: ContractData = {
    abi: JSON.stringify(compiledContract.abi),
    bytecode: compiledContract.bytecode,
  };

  return {
    id,
    contract,
  };
}

async function runAllSolidityTests(
  artifacts: Artifact[],
  testSuites: ArtifactId[],
  configArgs: SolidityTestRunnerConfigArgs
): Promise<SuiteResult[]> {
  return new Promise((resolve, reject) => {
    const resultsFromCallback: SuiteResult[] = [];

    runSolidityTests(
      artifacts,
      testSuites,
      configArgs,
      (suiteResult: SuiteResult) => {
        resultsFromCallback.push(suiteResult);
        if (resultsFromCallback.length === artifacts.length) {
          resolve(resultsFromCallback);
        }
      },
      reject
    );
  });
}
