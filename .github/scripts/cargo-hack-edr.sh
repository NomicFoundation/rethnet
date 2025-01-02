#!/bin/bash

set -e

# Execute cargo hack only for EDR crates
for dir in crates/edr_*/ ; do
    if [ -d "$dir" ]; then
      pushd "$dir" > /dev/null 
      cargo hack check --feature-powerset --exclude-no-default-features --no-dev-deps 
      popd > /dev/null
    fi
done

