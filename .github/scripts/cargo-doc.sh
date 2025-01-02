#!/bin/bash

set -e

# For EDR crates, test that docs build and they don't have warnings
for dir in crates/edr_*/ ; do
    if [ -d "$dir" ]; then
      pushd "$dir" > /dev/null
      RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
      popd > /dev/null
    fi
done

# For Foundry crates, only test that docs build and allow linking to private items
for dir in crates/foundry/* ; do
    if [ -d "$dir" ]; then
      pushd "$dir" > /dev/null
      cargo doc --all-features --no-deps --document-private-items
      popd > /dev/null
    fi
done

