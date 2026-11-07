#!/usr/bin/env bash
# Separately bounded, resumable build of PyTorch3D e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31.
# This stage does not share the FaceBoxes / nvdiffrast timeouts.
# A timeout leaves the persistent source tree in place so ninja can resume.
# It does not prove the commit is incompatible.
set -euo pipefail

COMMIT=e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CACHE=${LAM_COMPAT_CACHE:-"$HOME/.cache/lam-blackwell"}
REPORT=${LAM_REPORT_DIR:-"$CACHE/report"}
SRC=${LAM_PYTORCH3D_SRC:-"$CACHE/src/pytorch3d-$COMMIT"}
PY=${LAM_PYTHON:-python3}
if [[ -n "${LAM_VENV:-}" && -x "$LAM_VENV/bin/python" ]]; then
  PY="$LAM_VENV/bin/python"
fi
TIMEOUT=${PYTORCH3D_BUILD_TIMEOUT:-1800}
MAX_JOBS=${MAX_JOBS:-4}
export PYTHONPATH="$ROOT/scripts/compat${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REPORT" "$CACHE/src"

RESULT="$REPORT/pytorch3d_build.json"
write_result() {
  "$PY" - "$RESULT" "$@" << 'PY'
import json, sys
path, status = sys.argv[1], sys.argv[2]
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
doc = {
    "component": "pytorch3d",
    "commit": "e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31",
    "status": status,
    "kernel": "UNTESTED",
    "measured_g4_timeout_seconds": 160.004,
    "measured_g4_result": "BUILD_TIMEOUT",
    "measured_g4_note": "2026-11-06 160s cap was an artificial timeout. 2026-11-07 on torch 2.14.1+cu130 the build failed in 3.893s because setup.py passed -std=c++17 and ATen requires C++20.",
    "measured_g4_cpp_failure_seconds": 3.893,
    "measured_g4_cpp_failure_torch": "2.14.1+cu130",
    "proposed_cold_budget_seconds": 1800,
    "proposed_budget_basis": "31 .cu + 36 .cpp translation units, one CUDA arch, ninja, persistent tree. Estimate, not a completed measurement.",
    "tested_successful_build_seconds": None,
}
doc.update(extra)
open(path, "w").write(json.dumps(doc, indent=2) + "\n")
print(json.dumps(doc, indent=2))
PY
}

if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  write_result UNTESTED '{"reason":"torch is not importable in this interpreter","nvcc":"not consulted","refused_cpu_only_substitute":true}'
  exit 0
fi

CUDA_OK=$("$PY" - << 'PY'
import shutil, torch
print("1" if torch.cuda.is_available() else "0")
print(torch.__version__)
print(torch.version.cuda or "")
PY
)
mapfile -t CUDA_LINES <<< "$CUDA_OK"
HAS_CUDA=${CUDA_LINES[0]}
TORCH_VER=${CUDA_LINES[1]}
TORCH_CUDA=${CUDA_LINES[2]}
NVCC_BIN=$(command -v nvcc || true)

if [[ "$HAS_CUDA" != "1" && "${LAM_PYTORCH3D_FORCE_CUDA:-0}" != "1" ]]; then
  write_result UNTESTED "$(printf '{"reason":"torch.cuda.is_available() is false and LAM_PYTORCH3D_FORCE_CUDA is not set","torch":"%s","torch_cuda":"%s","nvcc":"%s","refused_cpu_only_substitute":true}' "$TORCH_VER" "$TORCH_CUDA" "$NVCC_BIN")"
  exit 0
fi

if [[ -z "$NVCC_BIN" ]]; then
  write_result UNTESTED "$(printf '{"reason":"nvcc is not on PATH; refusing to start a CUDA extension build","torch":"%s","torch_cuda":"%s"}' "$TORCH_VER" "$TORCH_CUDA")"
  exit 0
fi

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST=$("$PY" - << 'PY'
import torch
major, minor = torch.cuda.get_device_capability()
print(f"{major}.{minor}")
PY
  )
  export TORCH_CUDA_ARCH_LIST
fi
export MAX_JOBS
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$NVCC_BIN")")}

if [[ -z "${CUB_HOME:-}" ]]; then
  if [[ -f "$CUDA_HOME/include/cub/cub.cuh" ]]; then
    export CUB_HOME="$CUDA_HOME/include"
  elif [[ -f "$CUDA_HOME/include/cccl/cub/cub.cuh" ]]; then
    export CUB_HOME="$CUDA_HOME/include/cccl"
  fi
fi

if [[ ! -d "$SRC/.git" ]]; then
  git clone --filter=blob:none https://github.com/facebookresearch/pytorch3d.git "$SRC"
fi
git -C "$SRC" fetch --filter=blob:none origin "$COMMIT" || true
git -C "$SRC" checkout --force "$COMMIT"
# Do not git-clean: build/ and *.o are the resume cache.
# checkout restores upstream -std=c++17. Re-apply the C++20 patch every time.
patch -p1 --forward --batch --directory "$SRC" \
  < "$ROOT/scripts/compat/patches/pytorch3d-e4ad6e5-cpp20.patch"
if grep -n -- '-std=c++17' "$SRC/setup.py"; then
  echo "pytorch3d setup.py still requests -std=c++17" >&2
  exit 1
fi
grep -n -- '-std=c++20' "$SRC/setup.py" >"$REPORT/pytorch3d_std_flags.txt"
# Fail here if the absolute venv interpreter cannot run ninja. Do not fall
# through into distutils, which hides the missing executable and then compiles
# with the wrong standard.
"$PY" -c 'import lam_runtime; print(lam_runtime.require_ninja())' >"$REPORT/pytorch3d_ninja.txt"

"$PY" -m pip install -c "$ROOT/scripts/compat/constraints-blackwell-cu130.txt" \
  'iopath==0.1.10' 'fvcore==0.1.5.post20221221' \
  >"$REPORT/pytorch3d_deps.log" 2>&1

LOG="$REPORT/pytorch3d_build.log"
START=$("$PY" -c 'import time; print(time.time())')
set +e
(
  cd "$SRC"
  timeout --foreground "$TIMEOUT" "$PY" setup.py build_ext --inplace >"$LOG" 2>&1
)
RC=$?
set -e
END=$("$PY" -c 'import time; print(time.time())')
ELAPSED=$("$PY" -c "print(round($END - $START, 3))")

if [[ "$RC" -eq 0 ]]; then
  "$PY" -m pip install --no-build-isolation --no-deps -e "$SRC"
  write_result PASS "$(printf '{"phase":"build_ext_inplace","elapsed_seconds":%s,"timeout_seconds":%s,"timeout_kind":"proposed_not_previously_measured","torch":"%s","torch_cuda":"%s","TORCH_CUDA_ARCH_LIST":"%s","MAX_JOBS":"%s","CUB_HOME":"%s","source":"%s","log":"%s","kernel":"UNTESTED"}' "$ELAPSED" "$TIMEOUT" "$TORCH_VER" "$TORCH_CUDA" "$TORCH_CUDA_ARCH_LIST" "$MAX_JOBS" "${CUB_HOME:-}" "$SRC" "$LOG")"
  exit 0
fi

if [[ "$RC" -eq 124 ]]; then
  write_result BUILD_TIMEOUT_RESUMABLE "$(printf '{"reason":"stage timeout; ninja objects kept under the persistent tree","elapsed_seconds":%s,"timeout_seconds":%s,"timeout_kind":"proposed_not_previously_measured","torch":"%s","TORCH_CUDA_ARCH_LIST":"%s","MAX_JOBS":"%s","source":"%s","log":"%s"}' "$ELAPSED" "$TIMEOUT" "$TORCH_VER" "$TORCH_CUDA_ARCH_LIST" "$MAX_JOBS" "$SRC" "$LOG")"
  exit 124
fi

write_result FAIL "$(printf '{"reason":"setup.py build_ext failed","returncode":%s,"elapsed_seconds":%s,"log":"%s","source":"%s"}' "$RC" "$ELAPSED" "$LOG" "$SRC")"
exit "$RC"
