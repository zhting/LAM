#!/usr/bin/env python3
"""Reproduce the G4 ninja-discovery and C++17 failures, then check the fix.

CUDA compilation stays UNTESTED when nvcc is absent. Host g++ is executed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "compat"))

G4_NVCC = (
    "/usr/local/cuda/bin/nvcc -DWITH_CUDA -DTHRUST_IGNORE_CUB_VERSION_CHECK "
    "-I/content/pytorch3d/csrc -c /content/pytorch3d/csrc/ball_query/ball_query.cu "
    "-o /tmp/ball_query.o -D__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr "
    "--compiler-options '-fPIC' -DCUDA_HAS_FP16=1 -std=c++17 "
    "--device-entity-has-hidden-visibility=false -static-global-template-stub=false "
    "-gencode=arch=compute_120,code=sm_120"
)


def _which_cxx() -> str | None:
    for name in ("c++", "g++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import shlex

    import lam_runtime

    checks = []

    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    venv_python = os.environ.get("LAM_VENV_PYTHON", sys.executable)
    scripts = Path(sysconfig.get_path("scripts"))
    ninja_path = scripts / "ninja"
    purelib = Path(sysconfig.get_path("purelib"))
    for stale in (purelib / "lam_runtime.pth", purelib / "lam_runtime.py"):
        if stale.is_file() and stale.resolve() != (ROOT / "scripts" / "compat" / "lam_runtime.py").resolve():
            stale.unlink()
    bare_env = os.environ.copy()
    bare_env["PATH"] = "/usr/bin:/bin"
    bare_env.pop("LAM_NINJA", None)
    bare_env.pop("PYTHONPATH", None)
    bare_env.pop("PYTHONNOUSERSITE", None)

    if ninja_path.is_file():
        missing = subprocess.run(
            [venv_python, "-c", "import subprocess; subprocess.check_output(['ninja', '--version'])"],
            env=bare_env,
            capture_output=True,
            text=True,
        )
        reproduced = missing.returncode != 0
        add(
            "reproduce_ninja_invisible_without_activation",
            "PASS" if reproduced else "FAIL",
            f"returncode={missing.returncode} stderr={missing.stderr[-240:]}",
        )
        fixed_env = dict(bare_env)
        fixed_env["PYTHONPATH"] = str(ROOT / "scripts" / "compat")
        fixed = subprocess.run(
            [
                venv_python,
                "-c",
                "import lam_runtime,subprocess,sys;"
                "print(subprocess.check_output(['ninja','--version'],text=True).strip());"
                "child=subprocess.check_output([sys.executable,'-c','import os; print(os.environ[\"PATH\"])'],text=True);"
                "folder=__import__('os').path.dirname(lam_runtime.require_ninja());"
                "assert folder in child, child;"
                "print(lam_runtime.require_ninja())",
            ],
            env=fixed_env,
            capture_output=True,
            text=True,
        )
        add(
            "absolute_python_finds_ninja_and_propagates_path",
            "PASS" if fixed.returncode == 0 else "FAIL",
            (fixed.stdout + fixed.stderr)[-400:],
        )
        (purelib / "lam_runtime.py").write_text((ROOT / "scripts" / "compat" / "lam_runtime.py").read_text())
        (purelib / "lam_runtime.pth").write_text("import lam_runtime\n")
        pth_env = dict(bare_env)
        pth = subprocess.run(
            [venv_python, "-c", "import subprocess; print(subprocess.check_output(['ninja','--version'],text=True).strip())"],
            env=pth_env,
            capture_output=True,
            text=True,
        )
        add(
            "pth_hook_without_pythonpath",
            "PASS" if pth.returncode == 0 and pth.stdout.strip() else "FAIL",
            (pth.stdout + pth.stderr)[-300:],
        )
    else:
        add("reproduce_ninja_invisible_without_activation", "UNTESTED", f"no ninja at {ninja_path}")

    host = [
        "c++",
        "-std=c++17",
        "-I/usr/include",
        "-c",
        "aten_user.cpp",
        "-o",
        "aten_user.o",
    ]
    rewritten_host = lam_runtime.rewrite_compiler_command(host)
    add(
        "host_cxx17_rewritten_to_cxx20",
        "PASS" if "-std=c++20" in rewritten_host and "-std=c++17" not in rewritten_host else "FAIL",
        " ".join(rewritten_host),
    )
    nvcc_argv = shlex.split(G4_NVCC)
    rewritten_nvcc = lam_runtime.rewrite_compiler_command(nvcc_argv)
    nvcc_ok = (
        rewritten_nvcc[0].endswith("nvcc")
        and "-std=c++20" in rewritten_nvcc
        and "-std=c++17" not in rewritten_nvcc
        and "--device-entity-has-hidden-visibility=false" in rewritten_nvcc
    )
    add("g4_nvcc_argv_rewritten_to_cxx20", "PASS" if nvcc_ok else "FAIL", " ".join(rewritten_nvcc))
    doubled = lam_runtime.rewrite_compiler_command(["nvcc", "-std=c++17", "-std=c++20", "-c", "a.cu"])
    add(
        "nvcc_keeps_a_single_std_flag",
        "PASS" if [arg for arg in doubled if str(arg).startswith("-std=")] == ["-std=c++20"] else "FAIL",
        " ".join(map(str, doubled)),
    )

    cxx = _which_cxx()
    if cxx is None:
        add("host_gxx_cpp20_object", "UNTESTED", "c++ not on PATH")
    else:
        work = Path("/tmp/lam-cpp20-host")
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        source = work / "need_cpp20.cpp"
        source.write_text(
            textwrap.dedent(
                """\
                #if __cplusplus < 202002L
                #error C++20 required
                #endif
                int lam_cpp20_probe() { return 0; }
                """
            )
        )
        compiled = subprocess.run(
            [cxx, "-std=c++17", "-c", str(source), "-o", str(work / "need_cpp20.o")],
            capture_output=True,
            text=True,
        )
        add(
            "host_gxx_cpp20_object",
            "PASS" if compiled.returncode == 0 and (work / "need_cpp20.o").is_file() else "FAIL",
            (compiled.stderr or compiled.stdout)[-400:],
        )

    nvcc = shutil.which("nvcc")
    add(
        "nvcc_execution",
        "UNTESTED" if nvcc is None else "UNTESTED",
        "nvcc is not on this machine; argv rewrite was checked, the binary was not executed"
        if nvcc is None
        else "nvcc exists but CUDA object compilation is left UNTESTED in this offline repair",
    )

    failed = [item["name"] for item in checks if item["status"] == "FAIL"]
    result = {
        "test": "ninja discovery and C++ standard",
        "status": "FAIL" if failed else "PASS",
        "failed": failed,
        "python": sys.version,
        "checks": checks,
        "cuda_compile": "UNTESTED",
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
