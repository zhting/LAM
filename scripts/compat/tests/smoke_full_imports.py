#!/usr/bin/env python3
"""Weightless entry-point import smoke test.

PASS means the module or class imported, or a synthetic operator ran.
It does not mean FLAME fitting, checkpoint load, or LAM inference.
Constructors that read model files are not called; those cases are
BLOCKED_RESOURCE. CUDA operators are UNTESTED when there is no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

IMPORTS = [
    ("import_model_lam", "lam.models.modeling_lam", "ModelLAM"),
    ("import_flame_tracking_entry", "tools.flame_tracking_single_image", "FlameTrackingSingleImage"),
    ("import_global_tracker", "vhap.model.tracker", "GlobalTracker"),
    ("import_flame_head_class", "vhap.model.flame", "FlameHead"),
    ("import_faceboxes_class", "external.landmark_detection.FaceBoxesV2.faceboxes_detector", "FaceBoxesDetector"),
    ("import_vgghead_class", "external.vgghead_detector", "VGGHeadDetector"),
    ("import_stylematte_engine_class", "external.human_matting", "StyleMatteEngine"),
    ("import_stylematte_class", "external.human_matting.stylematte", "StyleMatte"),
    ("import_memeff_attention", "lam.models.encoders.dinov2.layers.attention", "MemEffAttention"),
    ("import_infer_module", "lam.runners.infer.lam", None),
]

BLOCKED = [
    ("construct_model_lam", "ModelLAM() builds Dinov2 with pretrained=True and reads human_model_path"),
    ("construct_dinov2_pretrained", "dinov2 hub downloads LVD142M via torch.hub.load_state_dict_from_url"),
    ("construct_flame_head", "FlameHead reads flame2023.pkl and landmark embeddings"),
    ("construct_flame_tracking", "FlameTrackingSingleImage loads FaceBoxes, VGGHead, matting, and alignment weights"),
    ("construct_global_tracker", "GlobalTracker constructs FlameHead"),
    ("construct_faceboxes", "FaceBoxesDetector torch.load's FaceBoxesV2.pth"),
    ("construct_vgghead", "VGGHeadDetector torch.load's vgg_heads_l.trcd"),
    ("construct_stylematte_engine", "StyleMatteEngine torch.load's stylematte_synth.pt"),
    ("load_lam_checkpoint", "safetensors / hf_hub checkpoint for LAM-20K step_045500"),
    ("flame_fit_or_inference", "no photo, no FLAME asset, no generate/fit call"),
]


def _child_env() -> dict:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def _run_code(code: str, seconds: int = 90) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=seconds,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-4000:],
    }


def _import_code(module: str, qualname: str | None) -> str:
    target = "mod" if qualname is None else "obj"
    get = "" if qualname is None else f"obj = getattr(mod, {qualname!r})\n"
    return textwrap.dedent(
        f"""
        import os, sys
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        sys.path.insert(0, {str(ROOT)!r})
        import torch
        def _blocked(*args, **kwargs):
            raise RuntimeError("BLOCKED_RESOURCE: torch hub download")
        torch.hub.load_state_dict_from_url = _blocked
        torch.hub.load = _blocked
        import importlib
        mod = importlib.import_module({module!r})
        {get}
        print({target})
        """
    )


def _status_from(result: dict) -> str:
    if result["returncode"] == 0:
        return "PASS"
    text = result["stdout"] + result["stderr"]
    if "BLOCKED_RESOURCE" in text:
        return "BLOCKED_RESOURCE"
    return "FAIL"


def _torchvision_case() -> dict:
    code = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(ROOT)!r})
        import torch
        import torchvision
        from torchvision.ops import nms
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]])
        scores = torch.tensor([0.9, 0.8])
        keep = nms(boxes, scores, 0.5)
        assert keep.numel() >= 1
        cuda = {{"status": "UNTESTED"}}
        if torch.cuda.is_available():
            keep_cuda = nms(boxes.cuda(), scores.cuda(), 0.5)
            torch.cuda.synchronize()
            cuda = {{"status": "PASS", "kept": int(keep_cuda.numel())}}
        print(json.dumps({{"cpu_kept": int(keep.numel()), "cuda": cuda, "torchvision": torchvision.__version__}}))
        """
    )
    try:
        result = _run_code(code)
    except subprocess.TimeoutExpired as exc:
        return {"name": "torchvision_nms", "status": "FAIL", "reason": f"timeout {exc}"}
    if result["returncode"] != 0:
        return {"name": "torchvision_nms", "status": "FAIL", "traceback": result["stderr"], "note": "ABI or missing torchvision"}
    payload = json.loads(result["stdout"].strip().splitlines()[-1])
    return {"name": "torchvision_nms", "status": "PASS", "result": payload, "scope": "operator_not_a_detector"}


def _xformers_case() -> dict:
    code = textwrap.dedent(
        """
        import json
        import torch
        from xformers.ops import memory_efficient_attention
        info = {"imported": True, "xformers_file": __import__("xformers").__file__}
        if not torch.cuda.is_available():
            info["cuda"] = "UNTESTED"
            print(json.dumps(info))
            raise SystemExit(0)
        q = torch.randn(1, 8, 4, 16, device="cuda", dtype=torch.float16)
        out = memory_efficient_attention(q, q, q)
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite xformers output")
        info["cuda"] = "PASS"
        info["shape"] = list(out.shape)
        print(json.dumps(info))
        """
    )
    try:
        result = _run_code(code, seconds=120)
    except subprocess.TimeoutExpired:
        return {"name": "xformers_memory_efficient_attention", "status": "FAIL", "reason": "timeout"}
    text = result["stderr"]
    if result["returncode"] != 0 and ("No module named 'xformers'" in text or "No module named \"xformers\"" in text):
        return {
            "name": "xformers_memory_efficient_attention",
            "status": "UNTESTED",
            "reason": "xformers is not installed. Upstream MemEffAttention falls back to math attention unless attn_bias is set. No source build was attempted.",
            "upstream_fallback": True,
        }
    if result["returncode"] != 0:
        return {"name": "xformers_memory_efficient_attention", "status": "FAIL", "traceback": text}
    payload = json.loads(result["stdout"].strip().splitlines()[-1])
    if payload.get("cuda") == "UNTESTED":
        return {"name": "xformers_memory_efficient_attention", "status": "UNTESTED", "result": payload, "reason": "no CUDA device; import succeeded"}
    return {"name": "xformers_memory_efficient_attention", "status": "PASS", "result": payload, "scope": "synthetic_operator_not_lam_inference"}


def _attention_forward() -> dict:
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        import torch
        from lam.models.encoders.dinov2.layers.attention import MemEffAttention, XFORMERS_AVAILABLE
        layer = MemEffAttention(dim=32, num_heads=4)
        y = layer(torch.randn(2, 5, 32))
        assert y.shape == (2, 5, 32) and torch.isfinite(y).all()
        print(int(XFORMERS_AVAILABLE))
        """
    )
    try:
        result = _run_code(code)
    except subprocess.TimeoutExpired:
        return {"name": "memeff_attention_random_init", "status": "FAIL", "reason": "timeout"}
    if result["returncode"] != 0:
        return {"name": "memeff_attention_random_init", "status": "FAIL", "traceback": result["stderr"]}
    return {
        "name": "memeff_attention_random_init",
        "status": "PASS",
        "xformers_available_flag": result["stdout"].strip().endswith("1"),
        "scope": "random_init_forward_not_a_checkpoint",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = []
    for name, module, qualname in IMPORTS:
        try:
            result = _run_code(_import_code(module, qualname))
        except subprocess.TimeoutExpired:
            cases.append({"name": name, "status": "FAIL", "reason": "timeout", "module": module})
            continue
        cases.append({
            "name": name,
            "status": _status_from(result),
            "module": module,
            "qualname": qualname,
            "traceback": None if result["returncode"] == 0 else result["stderr"],
        })
    cases.append(_attention_forward())
    cases.append(_torchvision_case())
    cases.append(_xformers_case())
    for name, reason in BLOCKED:
        cases.append({"name": name, "status": "BLOCKED_RESOURCE", "reason": reason, "invoked": False})

    failed = [case["name"] for case in cases if case["status"] == "FAIL"]
    blocked = [case["name"] for case in cases if case["status"] == "BLOCKED_RESOURCE"]
    untested = [case["name"] for case in cases if case["status"] == "UNTESTED"]
    passed = [case["name"] for case in cases if case["status"] == "PASS"]
    versions = {"python": sys.version}
    try:
        import torch
        versions["torch"] = torch.__version__
        versions["cuda"] = torch.version.cuda
        versions["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        versions["torch"] = None
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        versions["numpy"] = None
    document = {
        "overall": "FAIL" if failed else "PASS",
        "scope": "imports_and_synthetic_ops_not_fitting_or_inference",
        "passed": passed,
        "failed": failed,
        "blocked_resource": blocked,
        "untested": untested,
        "versions": versions,
        "cases": cases,
    }
    Path(args.out).write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"overall": document["overall"], "failed": failed, "untested": untested}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
