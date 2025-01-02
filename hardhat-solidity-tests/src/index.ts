import { task } from "hardhat/config";

task("test:solidity").setAction(async (_: any, hre: any) => {
  await hre.run("compile", { quiet: true });
  const { buildSolidityTestsInput, runAllSolidityTests } = await import(
    "@nomicfoundation/edr-helpers"
  );
  const { spec } = require("node:test/reporters");

  const specReporter = new spec();

  specReporter.pipe(process.stdout);

  let totalTests = 0;
  let failedTests = 0;

  const { artifacts, testSuiteIds } = await buildSolidityTestsInput(
    hre.artifacts,
    (artifact) => {
      const sourceName = artifact.id.source;
      const isTestArtifact =
        sourceName.endsWith(".t.sol") &&
        sourceName.startsWith("contracts/") &&
        !sourceName.startsWith("contracts/forge-std/") &&
        !sourceName.startsWith("contracts/ds-test/");

      return isTestArtifact;
    }
  );

  const config = {
    projectRoot: hre.config.paths.root,
  };

  await runAllSolidityTests(
    artifacts,
    testSuiteIds,
    config,
    (suiteResult, testResult) => {
      let name = suiteResult.id.name + " | " + testResult.name;
      if ("runs" in testResult?.kind) {
        name += ` (${testResult.kind.runs} runs)`;
      }

      const failed = testResult.status === "Failure";
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
  );

  console.log(`\n${totalTests} tests found, ${failedTests} failed`);

  if (failedTests > 0) {
    process.exit(1);
  }
  process.exit(0);
});
