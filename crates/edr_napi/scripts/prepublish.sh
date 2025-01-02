#!/usr/bin/env bash

set -e
set -o pipefail

pnpm napi prepublish -t npm --skip-gh-release

# temporarily commented to be able to publish in the @ignored org locally with verdaccio
# rename optionalDependencies key to dependencies
# jq 'with_entries(if .key == "optionalDependencies" then .key = "dependencies" else . end)' package.json | sponge package.json
