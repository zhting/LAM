#!/usr/bin/env python3
"""Synthetic CUDA kernel checks.

PASS is recorded only after a real kernel launch with finite results.
Missing torch, a missing GPU, or a package that was not built is UNTESTED.
Import success alone is never PASS. Exceptions from an actual launch are FAIL
and are not replaced with another implementation.
"""

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def find_spec(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def cuda_state():
    if not find_spec("torch"):
        return {"available": False, "reason": "torch is not installed", "torch": None, "device": None}
    import torch
    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "torch.cuda.is_available() is false",
            "torch": torch.__version__,
            "device": None,
        }
    return {
        "available": True,
        "reason": None,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def untested(name, reason):
    return {"name": name, "status": "UNTESTED", "reason": reason}


def fail(name, reason):
    return {"name": name, "status": "FAIL", "reason": reason}


def run_case(name, fn, state, requires_spec=None):
    if not state["available"]:
        return untested(name, state["reason"])
    if requires_spec and not find_spec(requires_spec):
        return untested(name, f"{requires_spec} is not installed")
    started = time.perf_counter()
    try:
        detail = fn()
    except Exception as exc:
        return fail(name, "".join(traceback.format_exception_only(type(exc), exc)).strip())
    detail = detail or {}
    detail.update({"name": name, "status": "PASS", "elapsed_seconds": round(time.perf_counter() - started, 6)})
    return detail


def test_matmul():
    import torch
    a = torch.ones((1024, 1024), device="cuda")
    b = torch.ones((1024, 1024), device="cuda")
    started = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    if not torch.isfinite(c).all() or c.shape != (1024, 1024):
        raise RuntimeError("matmul output is not a finite 1024x1024 tensor")
    return {"kind": "GPU_KERNEL", "checksum": float(c[0, 0].item()), "kernel_seconds": time.perf_counter() - started}


def test_gaussian():
    import inspect
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    means = torch.tensor([[0.0, 0.0, 2.0]], device="cuda", requires_grad=True)
    settings = dict(
        image_height=32,
        image_width=32,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=torch.eye(4, device="cuda"),
        sh_degree=0,
        campos=torch.zeros(3, device="cuda"),
        prefiltered=False,
        debug=False,
    )
    if "antialiasing" in inspect.signature(GaussianRasterizationSettings).parameters:
        settings["antialiasing"] = False
    raster = GaussianRasterizer(GaussianRasterizationSettings(**settings))
    image = raster(
        means3D=means,
        means2D=torch.zeros_like(means),
        shs=None,
        colors_precomp=torch.tensor([[1.0, 0.2, 0.1]], device="cuda"),
        opacities=torch.ones((1, 1), device="cuda") * 0.8,
        scales=torch.ones((1, 3), device="cuda") * 0.1,
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        cov3D_precomp=None,
    )[0]
    image.sum().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(image).all() or float(image.abs().sum()) <= 0 or means.grad is None:
        raise RuntimeError("gaussian raster did not produce a finite image and gradient")
    return {"kind": "GPU_KERNEL", "image_sum": float(image.sum().item())}


def test_knn():
    import torch
    from simple_knn._C import distCUDA2

    torch.manual_seed(123)
    points = torch.rand((32, 3), device="cuda")
    result = distCUDA2(points)
    torch.cuda.synchronize()
    distance = torch.cdist(points, points).square()
    distance.fill_diagonal_(float("inf"))
    reference = distance.topk(3, largest=False).values.mean(1)
    error = float((result - reference).abs().max().item())
    if not torch.isfinite(result).all() or error >= 1e-4:
        raise RuntimeError(f"knn max abs error {error}")
    return {"kind": "GPU_KERNEL", "max_abs_error": error}


def _triangle(winding):
    import torch
    front = [[-0.8, -0.8, 0.0, 1.0], [0.8, -0.8, 0.0, 1.0], [0.0, 0.8, 0.0, 1.0]]
    if winding == "back":
        front = [front[0], front[2], front[1]]
    vertices = torch.tensor([front], device="cuda", dtype=torch.float32, requires_grad=True)
    triangles = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    return vertices, triangles


def test_nvdiffrast():
    import torch
    import nvdiffrast.torch as dr

    context = dr.RasterizeCudaContext()
    front_v, triangles = _triangle("front")
    front, _ = dr.rasterize(context, front_v, triangles, resolution=[16, 16])
    front[..., :2].sum().backward()
    torch.cuda.synchronize()
    front_pixels = int((front[..., 3] > 0).sum().item())
    if front.shape != (1, 16, 16, 4) or front_pixels <= 0 or not torch.isfinite(front_v.grad).all():
        raise RuntimeError(f"front-facing raster failed, covered={front_pixels}")
    back_v, triangles = _triangle("back")
    back, _ = dr.rasterize(context, back_v, triangles, resolution=[16, 16])
    torch.cuda.synchronize()
    back_pixels = int((back[..., 3] > 0).sum().item())
    if back_pixels != 0:
        raise RuntimeError(
            f"back-facing triangle covered {back_pixels} pixels; "
            "ShenhanQian backface culling is not active"
        )
    return {"kind": "GPU_KERNEL", "front_pixels": front_pixels, "back_pixels": back_pixels, "culling": "checked"}


def test_pytorch3d():
    import torch
    from pytorch3d.ops import knn_points
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from pytorch3d.structures import Meshes

    points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device="cuda")
    knn = knn_points(points, points, K=1)
    if not torch.equal(knn.idx, torch.tensor([[[0], [1], [2]]], device="cuda")) or float(knn.dists.max()) != 0.0:
        raise RuntimeError("pytorch3d knn identity failed")
    verts = torch.tensor([[-0.8, -0.8, 1.0], [0.8, -0.8, 1.0], [0.0, 0.8, 1.0]], device="cuda", requires_grad=True)
    faces = torch.tensor([[0, 1, 2]], device="cuda")
    pix, zbuf, _, _ = rasterize_meshes(Meshes(verts=[verts], faces=[faces]), image_size=16, faces_per_pixel=1)
    zbuf[zbuf >= 0].sum().backward()
    torch.cuda.synchronize()
    covered = int((pix >= 0).sum().item())
    if covered <= 0 or not torch.isfinite(verts.grad).all():
        raise RuntimeError(f"pytorch3d mesh raster failed, covered={covered}")
    return {"kind": "GPU_KERNEL", "covered_pixels": covered}


def test_cpu_nms():
    script = ROOT / "scripts/compat/tests/test_faceboxes_cpu_nms.py"
    if not (ROOT / "external/landmark_detection/FaceBoxesV2/utils/nms").exists():
        return untested("faceboxes_cpu_nms", "FaceBoxes sources missing")
    extension = list((ROOT / "external/landmark_detection/FaceBoxesV2/utils/nms").glob("cpu_nms*.so"))
    if not extension:
        return untested("faceboxes_cpu_nms", "compiled cpu_nms extension is not present")
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nms.json"
        completed = subprocess.run([sys.executable, str(script), "--out", str(out)], capture_output=True, text=True)
        if completed.returncode != 0:
            return fail("faceboxes_cpu_nms", completed.stderr[-500:] or completed.stdout[-500:])
        payload = json.loads(out.read_text())
        payload["name"] = "faceboxes_cpu_nms"
        return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    state = cuda_state()
    cases = [
        run_case("base_cuda_matmul", test_matmul, state),
        run_case("diff_gaussian_rasterization", test_gaussian, state, "diff_gaussian_rasterization"),
        run_case("simple_knn", test_knn, state, "simple_knn"),
        run_case("nvdiffrast_backface_culling", test_nvdiffrast, state, "nvdiffrast"),
        run_case("pytorch3d_knn_and_raster", test_pytorch3d, state, "pytorch3d"),
        test_cpu_nms(),
    ]
    failed = [case["name"] for case in cases if case["status"] == "FAIL"]
    untested = [case["name"] for case in cases if case["status"] == "UNTESTED"]
    passed = [case["name"] for case in cases if case["status"] == "PASS"]
    if failed:
        status = "FAIL"
    elif not passed:
        status = "UNTESTED"
    elif untested:
        status = "PASS_WITH_UNTESTED"
    else:
        status = "PASS"
    result = {
        "status": status,
        "passed": passed,
        "untested": untested,
        "failed": failed,
        "gpu_device": state.get("device"),
        "torch": state.get("torch"),
        "gpu_suite": "UNTESTED" if not state["available"] else "EXECUTED",
        "note": "UNTESTED is not a pass. GPU PASS requires a real kernel launch.",
        "cases": cases,
        "python": sys.version,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
