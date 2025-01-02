// @ts-check

/** @type {import("syncpack").RcFile} */
const config = {
  versionGroups: [
    // smock only works with ethers v5
    {
      packages: ["hardhat-edr-smock-test"],
      dependencies: ["ethers"],
      // latest ethers v5 version
      pinVersion: "5.7.2",
    },
    {
      packages: ["**"],
      dependencies: ["@ignored/edr", "@nomicfoundation/edr-helpers", "hardhat-solidity-tests"],
      dependencyTypes: ["local"]
    },
  ],
  semverGroups: [
    {
      dependencies: ["typescript"],
      range: "~",
    },
  ],
};

module.exports = config;
