import { SuiteResult, TestSuite } from "@nomicfoundation/edr";
const { task } = require("hardhat/config");

task("test:solidity").setAction(async (_: any, hre: any) => {
  await hre.run("compile", { quiet: true });
  const { runSolidityTests } = await import("@nomicfoundation/edr");
  const { spec } = require("node:test/reporters");

  const specReporter = new spec();

  specReporter.pipe(process.stdout);

  let totalTests = 0;
  let failedTests = 0;

  const tests: TestSuite[] = [];
  const fqns = await hre.artifacts.getAllFullyQualifiedNames();

  for (const fqn of fqns) {
    const sourceName = fqn.split(":")[0];
    const isTestFile =
      sourceName.endsWith(".t.sol") &&
      sourceName.startsWith("contracts/") &&
      !sourceName.startsWith("contracts/forge-std/") &&
      !sourceName.startsWith("contracts/ds-test/");
    if (!isTestFile) {
      continue;
    }

    const artifact = hre.artifacts.readArtifactSync(fqn);

    const buildInfo = hre.artifacts.getBuildInfoSync(fqn);

    const test = {
      id: {
        artifactCachePath: hre.config.paths.cache,
        name: artifact.contractName,
        solcVersion: buildInfo.solcVersion,
        source: artifact.sourceName,
      },
      contract: {
        abi: JSON.stringify(artifact.abi),
        bytecode: artifact.bytecode,
        libsToDeploy: [],
        libraries: [],
      },
    };

    tests.push(test);
  }

  await new Promise<void>((resolve) => {
    const gasReport = false;

    runSolidityTests(tests, gasReport, (suiteResult: SuiteResult) => {
      for (const testResult of suiteResult.testResults) {
        let name = suiteResult.name + " | " + testResult.name;
        if ("runs" in testResult?.kind) {
          name += ` (${testResult.kind.runs} runs)`;
        }

        let failed = testResult.status === "Failure";
        totalTests++;
        if (failed) {
          failedTests++;
        }

        specReporter.write({
          type: failed ? "test:fail" : "test:pass",
          data: {
            name,
          },
        });
      }

      if (totalTests === tests.length) {
        resolve();
      }
    });
  });

  console.log(`\n${totalTests} tests found, ${failedTests} failed`);

  if (failedTests > 0) {
    process.exit(1);
  }
  process.exit(0);
});