#!/usr/bin/env python3
"""Static checks that do not compile CUDA and do not download weights."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    pyx = (ROOT / "external/landmark_detection/FaceBoxesV2/utils/nms/cpu_nms.pyx").read_text()
    check("pyx_no_numpy_int", "dtype=np.int)" not in pyx and "np.int_t" not in pyx, "cpu_nms.pyx")
    check("pyx_uses_intp", "np.intp_t" in pyx and "dtype=np.intp" in pyx, "argsort index dtype")
    build_py = (ROOT / "external/landmark_detection/FaceBoxesV2/utils/build.py").read_text()
    check("build_forces_cython", "self.force = True" in build_py, "build.py")
    make_sh = (ROOT / "external/landmark_detection/FaceBoxesV2/utils/make.sh").read_text()
    check("make_deletes_stale_c", "rm -f nms/cpu_nms.c" in make_sh, "make.sh")
    import subprocess
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", "external/landmark_detection/FaceBoxesV2/utils/nms/cpu_nms.c"],
        cwd=ROOT,
        text=True,
    ).strip()
    check("stale_cpu_nms_c_not_tracked", tracked == "", tracked or "untracked")
    generated = ROOT / "external/landmark_detection/FaceBoxesV2/utils/nms/cpu_nms.c"
    if generated.exists():
        check("generated_c_has_no_struct_subarray", "->subarray" not in generated.read_text(errors="replace"), str(generated))
    else:
        checks.append({"name": "generated_c_has_no_struct_subarray", "status": "UNTESTED", "detail": "cpu_nms.c not generated yet"})

    patch = (ROOT / "scripts/compat/patches/nvdiffrast-22718580-plugin-load.patch").read_text()
    check("patch_only_ops", patch.count("diff --git ") == 1 and "a/nvdiffrast/torch/ops.py" in patch, "patch paths")
    check("patch_publishes_module", "sys.modules[plugin_name] = module" in patch, "loader")
    check("patch_does_not_swallow", "except Exception" not in patch, "loader")
    check("patch_does_not_touch_culling", "EnableBackfaceCulling" not in patch and "rasterize_gl.cpp" not in patch, "culling sources unchanged")
    check("patch_publishes_ninja", "_ensure_ninja_on_subprocess_path()" in patch and "LAM_NINJA" in patch, "ninja discovery")
    cpp = (ROOT / "scripts/compat/patches/pytorch3d-e4ad6e5-cpp20.patch").read_text()
    check("pytorch3d_patch_is_setup_only", cpp.count("diff --git ") == 1 and "a/setup.py" in cpp, "cpp20 patch")
    check("pytorch3d_patch_upgrades_both_standards", cpp.count("-std=c++20") >= 2 and "-std=c++17" in cpp, "host and nvcc")
    wrapper = (ROOT / "external/landmark_detection/FaceBoxesV2/utils/nms_wrapper.py").read_text()
    check("tracker_still_calls_cpu_nms", "return cpu_nms(dets, thresh)" in wrapper, "no fake detector")

    nvd_src = os.environ.get("LAM_NVDIFFRAST_SRC", "")
    if nvd_src:
        ops = Path(nvd_src, "nvdiffrast/torch/ops.py").read_text()
        raster = Path(nvd_src, "nvdiffrast/torch/torch_rasterize.cpp").read_text()
        gl = Path(nvd_src, "nvdiffrast/common/rasterize_gl.cpp").read_text()
        check("patched_tree_has_loader", "LAM_COMPAT_PLUGIN_LOAD" in ops, nvd_src)
        check(
            "patched_tree_keeps_cuda_culling",
            "RenderModeFlag_EnableBackfaceCulling" in raster,
            "torch_rasterize.cpp",
        )
        check("patched_tree_keeps_gl_culling", "glEnable(GL_CULL_FACE)" in gl, "rasterize_gl.cpp")
    else:
        checks.append({"name": "patched_tree_culling", "status": "UNTESTED", "detail": "LAM_NVDIFFRAST_SRC unset"})

    pt_src = os.environ.get("LAM_PYTORCH3D_SRC", "")
    if pt_src:
        setup = Path(pt_src, "setup.py").read_text()
        check(
            "pytorch3d_pin_has_cuda13_flags",
            "--device-entity-has-hidden-visibility=false" in setup
            and "-static-global-template-stub=false" in setup,
            pt_src,
        )
    else:
        checks.append({"name": "pytorch3d_cuda13_flags", "status": "UNTESTED", "detail": "LAM_PYTORCH3D_SRC unset"})

    failed = [item for item in checks if item["status"] == "FAIL"]
    result = {
        "test": "static compatibility contracts",
        "status": "FAIL" if failed else "PASS",
        "checks": checks,
        "python": sys.version,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
