#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python}" -m worldreforge infer --checkpoint models/Cosmos3-Nano "$@"
