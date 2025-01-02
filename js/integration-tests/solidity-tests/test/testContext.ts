import {
  Artifact,
  ArtifactId,
  FuzzConfigArgs,
  InvariantConfigArgs,
  type SolidityTestRunnerConfigArgs,
} from "@ignored/edr";
import {
  buildSolidityTestsInput,
  runAllSolidityTests,
} from "@nomicfoundation/edr-helpers";
import hre from "hardhat";

export class TestContext {
  readonly rpcUrl = process.env.ALCHEMY_URL;
  readonly rpcCachePath: string = "./edr-cache";
  readonly fuzzFailuresPersistDir: string = "./edr-cache/fuzz";
  readonly invariantFailuresPersistDir: string = "./edr-cache/invariant";
  readonly artifacts: Artifact[];
  readonly testSuiteIds: ArtifactId[];

  private constructor(artifacts: Artifact[], testSuiteIds: ArtifactId[]) {
    this.artifacts = artifacts;
    this.testSuiteIds = testSuiteIds;
  }

  static async setup(): Promise<TestContext> {
    const results = await buildSolidityTestsInput(hre.artifacts);
    return new TestContext(results.artifacts, results.testSuiteIds);
  }

  defaultConfig(): SolidityTestRunnerConfigArgs {
    return {
      projectRoot: hre.config.paths.root,
      rpcCachePath: this.rpcCachePath,
    };
  }

  async runTestsWithStats(
    contractName: string,
    config?: Omit<SolidityTestRunnerConfigArgs, "projectRoot">
  ): Promise<SolidityTestsRunResult> {
    let totalTests = 0;
    let failedTests = 0;

    let testContracts = this.matchingTest(contractName);
    if (testContracts.length === 0) {
      throw new Error(`No matching test contract found for ${contractName}`);
    }

    const suiteResults = await runAllSolidityTests(
      this.artifacts,
      testContracts,
      {
        ...this.defaultConfig(),
        ...config,
      }
    );

    for (const suiteResult of suiteResults) {
      for (const testResult of suiteResult.testResults) {
        let failed = testResult.status === "Failure";
        totalTests++;
        if (failed) {
          failedTests++;
        }
      }
    }
    return { totalTests, failedTests };
  }

  matchingTest(contractName: string): ArtifactId[] {
    return this.matchingTests(new Set([contractName]));
  }
  matchingTests(testContractNames: Set<string>): ArtifactId[] {
    return this.testSuiteIds.filter((testSuiteId) => {
      return testContractNames.has(testSuiteId.name);
    });
  }
}

interface SolidityTestsRunResult {
  totalTests: number;
  failedTests: number;
}
