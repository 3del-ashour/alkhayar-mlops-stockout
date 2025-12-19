#!/usr/bin/env bash
set -euo pipefail
ruff check src serving tests
black --check src serving tests
