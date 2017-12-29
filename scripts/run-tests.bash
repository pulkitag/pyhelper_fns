#!/bin/bash

# propagate error
set -e

# cd to repo root
cd "${0%/*}/.."

# run tests
nose2
