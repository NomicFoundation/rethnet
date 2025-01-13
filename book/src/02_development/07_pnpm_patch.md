# `pnpm patch` a dependency

To modify/patch the code of a TypeScript dependency, you can use `pnpm patch`. E.g. to modify the `hardhat` dependency of the `hardhat-tests` folder, run:

```bash
pnpm patch hardhat@2.22.15
```

This will output something like:

```bash
Patch: You can now edit the package at:

  /workspaces/edr/node_modules/.pnpm_patches/hardhat@2.22.15

To commit your changes, run:

  pnpm patch-commit '/workspaces/edr/node_modules/.pnpm_patches/hardhat@2.22.15'
```

When editing the package, make sure to overwite both the source files in `node_modules/.pnpm_patches/hardhat@2.22.15/src/` and the generated files in `node_modules/.pnpm_patches/hardhat@2.22.15/`.

After you've edited the package at the listed directory and committed it using the `pnpm patch-commit` command, a `hardhat@2.22.15.patch` file will be created in the `patches/` directory. The top-level `package.json` file will also be updated with an entry looking something like this:

```bash
    "patchedDependencies": {
      "@defi-wonderland/smock@2.4.0": "patches/@defi-wonderland__smock@2.4.0.patch",
      "hardhat@2.22.15": "patches/hardhat@2.22.15.patch"
    }
```

Each patched dependency has an entry.

> BEWARE: sometimes the automated `pnpm patch-commit` tooling overwrites existing patched dependencies, so make sure that all entries are present.
