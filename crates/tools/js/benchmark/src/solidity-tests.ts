/*
Baseline

Source: https://github.com/NomicFoundation/forge-std/tree/js-benchmark-config

Foundry version: foundryup --commit 0a5b22f07

Commands:

forge test --fuzz-seed 0x1234567890123456789012345678901234567890 --no-match-test "test_ChainBubbleUp()|test_DeriveRememberKey()"
forge test --fuzz-seed 0x1234567890123456789012345678901234567890 --match-contract "StdCheatsTest"
forge test --fuzz-seed 0x1234567890123456789012345678901234567890 --match-contract "StdCheatsForkTest"
forge test --fuzz-seed 0x1234567890123456789012345678901234567890 --match-contract "StdMathTest"
forge test --fuzz-seed 0x1234567890123456789012345678901234567890 --match-contract "StdStorageTest"
forge test --fuzz-seed 0x1234567890123456789012345678901234567890 --match-contract "StdUtilsForkTest"
 */

import fs from "fs";
import { execSync } from "child_process";
import path from "path";
import simpleGit from "simple-git";
import { runAllSolidityTests } from "@nomicfoundation/edr-helpers";
import {
  SolidityTestRunnerConfigArgs,
  FsAccessPermission,
  Artifact,
  ArtifactId,
  ContractData,
} from "@ignored/edr";
import TOML from "smol-toml";

// This is automatically cached in CI
const RPC_CACHE_PATH = "./edr-cache";

// Hack: since EDR currently doesn't support filtering certain tests in test suites, we run them, but ignore their failures.
const EXCLUDED_TESTS = new Set([
  // This relies on environment variable interpolation in the `rpcEndpoints` config which is not supported by EDR.
  "test_ChainBubbleUp()",
  // This relies on the `deriveKey` and `rememberKey` cheatcodes which are not supported by EDR.
  "test_DeriveRememberKey()",
]);

// Total run for all test suites in the  `forge-std` repo
const TOTAL_NAME = "Total";
const TOTAL_EXPECTED_RESULTS = 15;

const DEFAULT_SAMPLES = 5;

// Map of test suites to benchmark individually to number of samples (how many times to run the test suite)
const TEST_SUITES = {
  [TOTAL_NAME]: DEFAULT_SAMPLES,
  StdCheatsTest: DEFAULT_SAMPLES,
  StdCheatsForkTest: 15,
  StdMathTest: 9,
  StdStorageTest: DEFAULT_SAMPLES,
  StdUtilsForkTest: 15,
};

const REPO_DIR = "forge-std";
const REPO_URL = "https://github.com/NomicFoundation/forge-std.git";
const BRANCH_NAME = "js-benchmark-config";

export async function setupForgeStdRepo() {
  const repoPath = path.join(__dirname, REPO_DIR);
  // Ensure directory exists
  if (!fs.existsSync(repoPath)) {
    await simpleGit().clone(REPO_URL, repoPath);
  }

  const git = simpleGit(repoPath);
  await git.fetch();
  await git.checkout(BRANCH_NAME);
  await git.pull();

  // Run npx hardhat compile
  execSync("npx hardhat compile", {
    cwd: repoPath,
    // Spawn child sharing only stderr.
    stdio: ["pipe", "pipe", process.stderr],
  });

  return repoPath;
}

/// Run Forge Standard Library tests and report to stdout
export async function runForgeStdTests(forgeStdRepoPath: string) {
  const artifactsDir = path.join(forgeStdRepoPath, "artifacts");
  const hardhatConfig = require(
    path.join(forgeStdRepoPath, "hardhat.config.js")
  );

  const { artifacts, testSuiteIds } = loadArtifacts(
    artifactsDir,
    hardhatConfig
  );

  const config = getConfig(forgeStdRepoPath);
  const allResults = [];
  const runs = new Map<string, number[]>();
  const recordRun = recordTime.bind(null, runs);

  for (const [name, samples] of Object.entries(TEST_SUITES)) {
    for (let i = 0; i < samples; i++) {
      let ids = testSuiteIds;
      if (name !== TOTAL_NAME) {
        ids = ids.filter((id) => id.name === name);
      }

      const start = performance.now();
      const results = await runAllSolidityTests(artifacts, ids, config);
      const elapsed = performance.now() - start;

      const expectedResults = name === TOTAL_NAME ? TOTAL_EXPECTED_RESULTS : 1;
      if (results.length !== expectedResults) {
        throw new Error(
          `Expected ${expectedResults} results for ${name}, got ${results.length}`
        );
      }

      const failed = new Set();
      for (const res of results) {
        for (const r of res.testResults) {
          if (r.status !== "Success" && !EXCLUDED_TESTS.has(r.name)) {
            failed.add(
              `${res.id.name} ${r.name} ${r.status} reason:\n${r.reason}`
            );
          }
        }
      }
      if (failed.size !== 0) {
        console.error(failed);
        throw new Error(`Some tests failed`);
      }

      // Log to stderr so that it doesn't pollute stdout where we write the results
      console.error(
        `elapsed (s) on run ${i + 1}/${samples} for ${name}: ${displaySec(elapsed)}`
      );

      if (name === TOTAL_NAME) {
        recordRun(TOTAL_NAME, elapsed);
      } else {
        if (results.length !== 1) {
          throw new Error(
            `Expected 1 result for ${name}, got ${results.length}`
          );
        }
        recordRun(results[0].id.name, elapsed);
      }

      // Hold on to all results to prevent GC from interfering with the benchmark
      allResults.push(results);
    }
  }

  const measurements = getMeasurements(runs);

  // Log info to stderr so that it doesn't pollute stdout where we write the results
  console.error("median total elapsed (s)", displaySec(measurements[0].value));

  console.log(JSON.stringify(measurements));
}

function getConfig(forgeStdRepoPath: string): SolidityTestRunnerConfigArgs {
  const foundryTomlPath = path.join(forgeStdRepoPath, "foundry.toml");

  if (!fs.existsSync(foundryTomlPath)) {
    throw new Error(`Get config failed: could not find ${foundryTomlPath}`);
  }
  const foundryToml = fs.readFileSync(foundryTomlPath, "utf8");
  const foundryTomlConfig = TOML.parse(foundryToml);

  const rpcEndpoints = foundryTomlConfig.rpc_endpoints as Record<
    string,
    string
  >;

  return {
    projectRoot: forgeStdRepoPath,
    rpcCachePath: RPC_CACHE_PATH,
    fsPermissions: [
      { path: forgeStdRepoPath, access: FsAccessPermission.ReadWrite },
    ],
    testFail: true,
    rpcEndpoints,
    fuzz: {
      // Used to ensure deterministic fuzz execution
      seed: "0x1234567890123456789012345678901234567890",
    },
  };
}

function getMeasurements(runs: Map<string, number[]>) {
  const results: Array<{ name: string; unit: string; value: number }> = [];

  const total = runs.get(TOTAL_NAME)!;
  results.push({ name: TOTAL_NAME, unit: "ms", value: medianMs(total) });
  runs.delete(TOTAL_NAME);

  const testSuiteNames = Array.from(runs.keys());
  testSuiteNames.sort();

  for (const name of testSuiteNames) {
    const value = medianMs(runs.get(name)!);
    results.push({ name, unit: "ms", value });
  }

  return results;
}

function medianMs(values: number[]) {
  if (values.length % 2 === 0) {
    throw new Error("Expected odd number of values");
  }
  values.sort((a, b) => a - b);
  const half = Math.floor(values.length / 2);
  // Round to get rid of decimal milliseconds
  return Math.round(values[half]);
}

function recordTime(
  runs: Map<string, number[]>,
  name: string,
  elapsed: number
) {
  let measurements = runs.get(name);
  if (measurements === undefined) {
    measurements = [];
    runs.set(name, measurements);
  }
  measurements.push(elapsed);
}

function displaySec(delta: number) {
  const sec = delta / 1000;
  return Math.round(sec * 100) / 100;
}

// Load contracts built with Hardhat
function loadArtifacts(
  artifactsDir: string,
  hardhatConfig: { solidity: { version: string } }
) {
  const artifacts: Artifact[] = [];
  const testSuiteIds: ArtifactId[] = [];

  for (const artifactPath of listFilesRecursively(artifactsDir)) {
    // Not a contract artifact file
    if (
      !artifactPath.endsWith(".json") ||
      artifactPath.endsWith(".dbg.json") ||
      artifactPath.includes("build-info")
    ) {
      continue;
    }
    const compiledContract = require(artifactPath);

    const id: ArtifactId = {
      name: compiledContract.contractName,
      solcVersion: hardhatConfig.solidity.version,
      source: compiledContract.sourceName,
    };

    if (isTestSuite(compiledContract)) {
      testSuiteIds.push(id);
    }

    const contract: ContractData = {
      abi: JSON.stringify(compiledContract.abi),
      bytecode: compiledContract.bytecode,
      deployedBytecode: compiledContract.deployedBytecode,
    };

    artifacts.push({ id, contract });
  }

  return {
    artifacts,
    testSuiteIds,
  };
}

function listFilesRecursively(dir: string, fileList: string[] = []): string[] {
  const files = fs.readdirSync(dir);

  files.forEach((file) => {
    const filePath = path.join(dir, file);
    if (fs.statSync(filePath).isDirectory()) {
      listFilesRecursively(filePath, fileList);
    } else {
      fileList.push(filePath);
    }
  });

  return fileList;
}

function isTestSuite(artifact: {
  abi: undefined | [{ type: string; name: string }];
}) {
  return (
    artifact.abi !== undefined &&
    artifact.abi.some(
      (item: { type: string; name: string }) =>
        item.type === "function" && item.name.startsWith("test")
    )
  );
}
