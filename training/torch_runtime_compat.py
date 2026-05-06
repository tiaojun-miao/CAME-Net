"""
torch_runtime_compat.py - Minimal runtime guards for fragile Torch environments.

Some Windows/Conda Torch installations ship with a broken torch.onnx exporter
stack that is imported indirectly through torch._dynamo when optimizers are
constructed. Disabling dynamo avoids that bad import path and keeps the project
usable without changing model behavior.
"""

from __future__ import annotations

import os
from typing import Dict


def configure_torch_runtime_compat() -> Dict[str, str]:
    os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    return {
        "TORCH_DISABLE_DYNAMO": os.environ["TORCH_DISABLE_DYNAMO"],
        "TORCHDYNAMO_DISABLE": os.environ["TORCHDYNAMO_DISABLE"],
    }
