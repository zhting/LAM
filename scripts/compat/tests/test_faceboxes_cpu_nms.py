#!/usr/bin/env python3
"""Exercise the compiled FaceBoxes CPU NMS extension. No GPU and no weights."""

import argparse
import importlib.machinery
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "external" / "landmark_detection" / "FaceBoxesV2" / "utils"
sys.path.insert(0, str(UTILS))


def _as_ints(values):
    return [int(v) for v in values]


def reference_nms(dets, thresh):
    sys.path.insert(0, str(UTILS))
    from nms.py_cpu_nms import py_cpu_nms
    return _as_ints(py_cpu_nms(np.array(dets, dtype=np.float32, copy=True), thresh))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    from nms.cpu_nms import cpu_nms, cpu_soft_nms

    module_file = Path(sys.modules["nms.cpu_nms"].__file__)
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    if not str(module_file).endswith(suffixes):
        raise SystemExit(f"cpu_nms resolved to {module_file}, not a compiled extension")

    generated = UTILS / "nms" / "cpu_nms.c"
    generated_text = generated.read_text(errors="replace") if generated.exists() else ""
    if "->subarray" in generated_text:
        raise SystemExit("regenerated cpu_nms.c still accesses PyArray_Descr.subarray")

    basic = np.array(
        [[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8], [20, 20, 30, 30, 0.7]],
        dtype=np.float32,
    )
    kept = _as_ints(cpu_nms(basic, 0.5))
    if kept != [0, 2]:
        raise SystemExit(f"diagnostic box set kept {kept}, expected [0, 2]")

    disjoint = np.array(
        [[0, 0, 1, 1, 0.2], [10, 10, 12, 12, 0.4], [20, 20, 21, 21, 0.3]],
        dtype=np.float32,
    )
    if _as_ints(cpu_nms(disjoint, 0.5)) != [1, 2, 0]:
        raise SystemExit("disjoint boxes were not kept in score order")

    rng = np.random.default_rng(1234)
    scores = np.linspace(0.05, 0.99, 48, dtype=np.float32)
    rng.shuffle(scores)
    random_boxes = np.concatenate(
        [rng.uniform(0, 64, size=(48, 4)).astype(np.float32), scores[:, None]],
        axis=1,
    )
    compiled = _as_ints(cpu_nms(random_boxes, 0.3))
    pure = reference_nms(random_boxes, 0.3)
    if compiled != pure:
        raise SystemExit(f"compiled NMS diverged from py_cpu_nms: {compiled} vs {pure}")

    soft_boxes = np.array(
        [[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [30, 30, 40, 40, 0.7]],
        dtype=np.float32,
    )
    soft_kept = _as_ints(cpu_soft_nms(soft_boxes.copy(), 0.5, 0.3, 0.001, 0))
    if soft_kept != [0, 1]:
        raise SystemExit(f"cpu_soft_nms kept {soft_kept}, expected [0, 1]")

    result = {
        "test": "FaceBoxes compiled CPU NMS",
        "status": "PASS",
        "kind": "COMPILED_CPU",
        "gpu": "UNTESTED",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "module_file": str(module_file),
        "generated_c_has_struct_subarray": False,
        "basic_kept": kept,
        "random_kept": compiled,
        "soft_nms_kept": soft_kept,
        "numpy": np.__version__,
        "python": sys.version,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
