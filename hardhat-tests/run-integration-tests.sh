#!/usr/bin/env bash

set -e

(cd .. && pnpm build)

cd integration
for i in *; do
  if [ -d "$i" ]; then
    echo "Running integration test: '$i'"
    cd $i
    pnpm hardhat test
    cd ..
  fi
done
