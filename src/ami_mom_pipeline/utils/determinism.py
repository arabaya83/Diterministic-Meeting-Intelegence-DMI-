from __future__ import annotations

import os
import random
import sys
from typing import Any


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_determinism(seed: int, strict: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "seed": seed,
        "strict_requested": bool(strict),
        "python": {
            "python_version": sys.version.split()[0],
            "hash_seed": str(seed),
        },
        "env": {},
        "libraries": {},
        "risks": [],
    }

    set_seed(seed)

    # Best-effort environment settings for deterministic behavior in common stacks.
    env_updates = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for k, v in env_updates.items():
        os.environ.setdefault(k, v)
    report["env"] = {k: os.environ.get(k) for k in sorted(env_updates)}

    # numpy (optional)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        report["libraries"]["numpy"] = {"available": True, "seed_applied": True, "version": getattr(np, "__version__", None)}
    except Exception as exc:
        report["libraries"]["numpy"] = {"available": False, "seed_applied": False, "error": f"{type(exc).__name__}: {exc}"}

    # torch (optional / best effort)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        torch_use_det_ok = True
        torch_use_det_err = None
        try:
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=not strict)
        except Exception as exc:
            torch_use_det_ok = False
            torch_use_det_err = f"{type(exc).__name__}: {exc}"
            report["risks"].append("torch_deterministic_algorithms_not_enforced")
        report["libraries"]["torch"] = {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)) if hasattr(torch, "backends") else None,
            "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)) if hasattr(torch, "backends") else None,
            "deterministic_algorithms_requested": True,
            "deterministic_algorithms_enforced": torch_use_det_ok,
            "deterministic_algorithms_error": torch_use_det_err,
        }
        if torch.cuda.is_available():
            report["risks"].append("gpu_execution_may_still_be_nondeterministic_for_some_kernels")
    except Exception as exc:
        report["libraries"]["torch"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    return report
