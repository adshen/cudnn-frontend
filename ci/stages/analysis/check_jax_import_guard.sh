#!/bin/bash
# Verify that cudnn-frontend installed with the [cutedsl] extra can be imported
# in an environment without torch (NGC JAX container). The cutedsl extra is
# framework-neutral by design: torch lives in a separate PEP 735 dependency
# group, and importing cudnn (including cudnn.jax) must never pull it in.
set -eux

# The check is only meaningful if torch is genuinely absent from the container.
if pip show torch > /dev/null 2>&1; then
    echo "ERROR: torch is installed in this container; the torch-free import check is invalid." >&2
    exit 1
fi

pip install -v .[cutedsl]

python - <<'EOF'
import sys

import cudnn

print("cudnn frontend:", cudnn.__version__)

import cudnn.jax  # noqa: F401  the JAX entry point must work without torch

torch_modules = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
assert not torch_modules, f"importing cudnn pulled in torch modules: {torch_modules}"

print("OK: cudnn imported with the [cutedsl] extra, torch-free.")
EOF
