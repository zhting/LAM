"""Resolve the import-stage pins before a GPU is allocated.

This does not install torch and does not claim a CUDA operator works.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import Version
except ImportError:  # a Python that only has the standalone packaging package
    from packaging.requirements import Requirement
    from packaging.version import Version

REQUIREMENTS = Path(__file__).resolve().parent / "constraints-import-cu130.txt"

# Requires-Dist verified against PyPI / cu130 wheels on 2026-11-07.
# torch is not installed here; 2.14.1+cu130 is the already-proven runtime pin.
CROSS_STAGE = {
    "moviepy": {
        "4.4.2-compatible": "decorator>=4.0.2,<5",
        "requires": [
            "decorator>=4.0.2,<5",
            "imageio>=2.5,<3",
            "imageio-ffmpeg>=0.2.0",
            "numpy>=1.17.3",
            "proglog<=1.0.0",
            "requests>=2.8.1",
            "tqdm>=4.11.2",
        ],
    },
    "transformers": {
        "requires": [
            "huggingface-hub<1.0,>=0.23.2",
            "numpy>=1.17",
            "tokenizers<0.21,>=0.20",
            "safetensors>=0.4.1",
        ],
    },
    "diffusers": {
        "requires": [
            "huggingface-hub>=0.23.2",
            "safetensors>=0.3.1",
            "numpy",
            "Pillow",
            "importlib-metadata",
        ],
    },
    "accelerate": {
        "requires": [
            "numpy>=1.17",
            "packaging>=20.0",
            "psutil",
            "pyyaml",
            "torch>=2.0.0",
            "huggingface-hub>=0.21.0",
            "safetensors>=0.4.3",
        ],
    },
    "torchvision": {
        "requires": ["numpy", "torch>=2.14.0", "pillow!=8.3.*,>=5.3.0"],
    },
    "xformers": {
        "requires": ["torch>=2.10", "numpy"],
    },
}

VIRTUAL_RUNTIME = {
    "torch": "2.14.1+cu130",
    "numpy": "2.4.6",
}


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path = REQUIREMENTS) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[canon(name)] = version
    return pins


def satisfied(requirement: str, provided: dict[str, str]) -> bool:
    req = Requirement(requirement)
    version = provided.get(canon(req.name))
    if version is None:
        return False
    if not req.specifier:
        return True
    return Version(version) in req.specifier


def cross_stage_gaps(pins: dict[str, str], runtime: dict[str, str] | None = None) -> list[str]:
    provided = {canon(name): version for name, version in pins.items()}
    provided.update({canon(name): version for name, version in (runtime or VIRTUAL_RUNTIME).items()})
    # Pillow is the distribution name of the pillow pin.
    if "pillow" in provided:
        provided.setdefault("pillow", provided["pillow"])
    gaps = []
    for package, info in CROSS_STAGE.items():
        for requirement in info["requires"]:
            if not satisfied(requirement, provided):
                gaps.append(f"{package} requires {requirement}; have {provided.get(Requirement(requirement).name)}")
    return gaps


def pip_dry_run(requirements: Path, constraints: list[str], python: str = sys.executable) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(constraints) + "\n")
        constraint_path = handle.name
    return subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "-r",
            str(requirements),
            "-c",
            constraint_path,
        ],
        capture_output=True,
        text=True,
    )


def broken_moviepy_resolution(python: str = sys.executable) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("decorator==5.1.1\nmoviepy==1.0.3\n")
        path = handle.name
    return subprocess.run(
        [python, "-m", "pip", "install", "--dry-run", "--ignore-installed", "-r", path],
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 13):
        raise SystemExit(f"FAIL_EARLY: preflight wants Python 3.13, this interpreter is {sys.version}")
    pins = read_pins()
    gaps = cross_stage_gaps(pins)
    broken = broken_moviepy_resolution()
    resolved = pip_dry_run(REQUIREMENTS, ["numpy==2.4.6"])
    report = {
        "python": sys.version,
        "decorator": pins.get("decorator"),
        "moviepy": pins.get("moviepy"),
        "diffusers": pins.get("diffusers") if "diffusers" in pins else "0.32.2 via import_plan",
        "broken_decorator_5_rejected": broken.returncode != 0,
        "broken_returncode": broken.returncode,
        "requirements_dry_run": "PASS" if resolved.returncode == 0 else "FAIL",
        "dry_run_returncode": resolved.returncode,
        "cross_stage_gaps": gaps,
        "cuda_operators": "UNTESTED",
        "torch_installed_by_preflight": False,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    log = Path(args.report).with_suffix(".log")
    log.write_text(broken.stdout + "\n" + broken.stderr + "\n---- dry-run ----\n" + resolved.stdout + "\n" + resolved.stderr)
    print(json.dumps({k: report[k] for k in ("broken_decorator_5_rejected", "requirements_dry_run", "cross_stage_gaps")}, indent=2))
    if broken.returncode == 0 or resolved.returncode != 0 or gaps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
