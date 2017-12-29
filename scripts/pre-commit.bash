#!/bin/bash

echo "Running pre-commit hook..."
./scripts/run-tests.bash

if [ $? -ne 0 ]; then
    echo "Tests failed!"
    exit 1
fi
