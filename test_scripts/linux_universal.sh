#!/bin/bash

ARCH=$(uname -m)
BINARY=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$ARCH" == "x86_64" ]]; then
    BINARY="iqtree3_intel"
elif [[ "$ARCH" == "aarch64" ]]; then
    BINARY="iqtree3_arm"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Run the appropriate binary
exec "${SCRIPT_DIR}/$BINARY" "$@"
