import type { Artifacts as HardhatArtifacts } from "hardhat/types";

import {
  ArtifactId,
  SuiteResult,
  runSolidityTests,
  Artifact,
  SolidityTestRunnerConfigArgs,
  TestResult,
} from "@ignored/edr";

/**
 * Run all the given solidity tests and returns the whole results after finishing.
 */
export async function runAllSolidityTests(
  artifacts: Artifact[],
  testSuites: ArtifactId[],
  configArgs: SolidityTestRunnerConfigArgs,
  testResultCallback: (
    suiteResult: SuiteResult,
    testResult: TestResult
  ) => void = () => {}
): Promise<SuiteResult[]> {
  return new Promise((resolve, reject) => {
    const resultsFromCallback: SuiteResult[] = [];

    runSolidityTests(
      artifacts,
      testSuites,
      configArgs,
      (suiteResult: SuiteResult) => {
        for (const testResult of suiteResult.testResults) {
          testResultCallback(suiteResult, testResult);
        }

        resultsFromCallback.push(suiteResult);
        if (resultsFromCallback.length === testSuites.length) {
          resolve(resultsFromCallback);
        }
      },
      reject
    );
  });
}

export async function buildSolidityTestsInput(
  hardhatArtifacts: HardhatArtifacts,
  isTestArtifact: (artifact: Artifact) => boolean = () => true
): Promise<{ artifacts: Artifact[]; testSuiteIds: ArtifactId[] }> {
  const fqns = await hardhatArtifacts.getAllFullyQualifiedNames();
  const artifacts: Artifact[] = [];
  const testSuiteIds: ArtifactId[] = [];

  for (const fqn of fqns) {
    const hardhatArtifact = hardhatArtifacts.readArtifactSync(fqn);
    const buildInfo = hardhatArtifacts.getBuildInfoSync(fqn);

    if (buildInfo === undefined) {
      throw new Error(`Build info not found for contract ${fqn}`);
    }

    const id = {
      name: hardhatArtifact.contractName,
      solcVersion: buildInfo.solcVersion,
      source: hardhatArtifact.sourceName,
    };

    const contract = {
      abi: JSON.stringify(hardhatArtifact.abi),
      bytecode: hardhatArtifact.bytecode,
      deployedBytecode: hardhatArtifact.deployedBytecode,
    };

    const artifact = { id, contract };
    artifacts.push(artifact);
    if (isTestArtifact(artifact)) {
      testSuiteIds.push(artifact.id);
    }
  }

  return { artifacts, testSuiteIds };
}
