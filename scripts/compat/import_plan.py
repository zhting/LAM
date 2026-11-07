"""Wheel plan for the weightless LAM import stage.

Pins are for torch 2.14.1+cu130 only. An unknown runtime torch fails
before any install, so pip cannot guess a replacement build.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Verified 2026-11-07 against https://download.pytorch.org/whl/cu130/
# torchvision-0.29.1+cu130-cp313 is a real cp313 manylinux wheel.
# xformers-0.0.35 is tagged py39-none; its _C.so links libtorch_cuda and
# libcudart.so.13 and does not link libpython, so cp313 can load it.
# There is no separate cp313 xformers build. Do not compile xformers.
ACCELERATE_PIN = "1.15.0"
# 0.32.2 still exports diffusers.utils.is_torch_version and accepts
# huggingface-hub 0.26.5 plus safetensors 0.4.5. 0.42.1 requires both newer.
DIFFUSERS_PIN = "0.32.2"

PAIRS = {
    "2.14.1+cu130": {
        "torchvision": "0.29.1+cu130",
        "xformers": "0.0.35",
        "index": "https://download.pytorch.org/whl/cu130",
        "python": "cp313",
        "platform": "manylinux_2_28_x86_64",
    }
}

def pure_pins() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "constraints-import-cu130.txt")
    pins = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                pins.append(line)
    return pins


def pair_for(torch_version: str) -> dict:
    pair = PAIRS.get(torch_version)
    if pair is None:
        known = ", ".join(PAIRS)
        raise SystemExit(
            f"FAIL_EARLY: runtime torch is {torch_version!r}. "
            f"This import stage only has a wheel pair for: {known}. "
            "Refusing to install torchvision or xformers, and refusing to change the installed torch."
        )
    return pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", required=True)
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "FAIL_EARLY: torch is not importable. Reuse the kernel-stage venv; do not pip install torch."
        ) from exc
    version = torch.__version__
    cuda = torch.version.cuda
    pair = pair_for(version)
    if cuda != "13.0":
        raise SystemExit(
            f"FAIL_EARLY: torch {version} reports CUDA {cuda!r}, expected 13.0. Not installing a different CUDA wheel."
        )
    plan = {
        "torch": version,
        "torch_cuda": cuda,
        "python": sys.version,
        "pair": pair,
        "pure_pins": pure_pins(),
        "no_deps": ["torchvision", "xformers", "accelerate", "diffusers"],
        "accelerate": ACCELERATE_PIN,
        "diffusers": DIFFUSERS_PIN,
        "pymcubes": {
            "spec": "pymcubes==0.1.6",
            "wheel_cp313": False,
            "timeout_seconds": 120,
            "note": "sdist only through cp312. G4 gcc may compile it; this offline clang image could not find cmath. Cap 120s.",
        },
        "not_installed": {
            "tensorflow": "requirements pin 2.12 has no cp313 wheel; iris/fdlite import is lazy and not on the default tracker import",
            "chumpy": "sdist only; needed to unpickle FLAME, which this stage does not do",
            "numpy_downgrade": "keep the installed NumPy 2.x",
        },
    }
    with open(args.write, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"torch": version, "torchvision": pair["torchvision"], "xformers": pair["xformers"]}))


if __name__ == "__main__":
    main()
