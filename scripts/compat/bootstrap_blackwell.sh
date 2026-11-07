#!/usr/bin/env bash
# Isolated Blackwell bootstrap for zhting/LAM.
# Reuses a system torch when one is importable. Does not install into the
# base environment and does not download model weights.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CACHE=${LAM_COMPAT_CACHE:-"$HOME/.cache/lam-blackwell"}
REPORT=${LAM_REPORT_DIR:-"$CACHE/report"}
VENV=${LAM_VENV:-"$ROOT/.venv-blackwell"}
BASE_PY=${LAM_PYTHON:-python3}
NV_COMMIT=22718580f24a313c429ba2c304794c264351f108
NV_SRC=${LAM_NVDIFFRAST_SRC:-"$CACHE/src/nvdiffrast-$NV_COMMIT"}
PATCH="$ROOT/scripts/compat/patches/nvdiffrast-22718580-plugin-load.patch"
CONSTRAINTS="$ROOT/scripts/compat/constraints-blackwell-cu130.txt"
mkdir -p "$CACHE/src" "$REPORT" "${TORCH_EXTENSIONS_DIR:-$CACHE/torch_extensions}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-4}"

log() { printf '%s\n' "$*" | tee -a "$REPORT/bootstrap.log"; }

create_venv() {
  local site_args=()
  if "$BASE_PY" -c 'import torch' >/dev/null 2>&1 || [[ "${LAM_REUSE_SYSTEM_TORCH:-0}" == "1" ]]; then
    site_args+=(--system-site-packages)
    log "venv will reuse base site-packages so the existing torch is visible"
  fi
  if [[ ! -x "$VENV/bin/python" ]]; then
    if ! "$BASE_PY" -m venv "${site_args[@]}" "$VENV" >"$REPORT/venv_setup.log" 2>&1; then
      log "default venv failed (often missing ensurepip); retrying --without-pip"
      "$BASE_PY" -m venv --without-pip "${site_args[@]}" "$VENV" >"$REPORT/venv_setup.log" 2>&1
    fi
  else
    log "reusing venv $VENV"
  fi
  if ! "$VENV/bin/python" -m pip --version >"$REPORT/venv_pip_setup.log" 2>&1; then
    log "bootstrapping pip into the venv only"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python" >>"$REPORT/venv_pip_setup.log" 2>&1
  fi
  "$VENV/bin/python" -m pip --version >>"$REPORT/venv_pip_setup.log"
}

create_venv
PY="$VENV/bin/python"
# Installs must go through the venv interpreter. Never `sudo pip` the base.
"$PY" -c 'import sys; assert sys.prefix != sys.base_prefix, sys.prefix'

log "installing pinned build tools from constraints"
"$PY" -m pip install -c "$CONSTRAINTS" numpy==2.4.6 'Cython==3.0.12' 'setuptools==80.10.2' 'ninja==1.13.2' \
  >"$REPORT/constraints_install.log" 2>&1
install_runtime() {
  local site
  site=$("$PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  cp "$ROOT/scripts/compat/lam_runtime.py" "$site/lam_runtime.py"
  printf '%s\n' 'import lam_runtime' > "$site/lam_runtime.pth"
  # Absolute interpreter, PATH deliberately without the venv bin directory.
  PATH="/usr/bin:/bin" "$PY" - << 'PY'
import lam_runtime, os, subprocess, sys
resolved = lam_runtime.require_ninja()
subprocess.check_output(["ninja", "--version"])
child = subprocess.check_output([sys.executable, "-c", "import os; print(os.environ['PATH'])"], text=True)
folder = os.path.dirname(resolved)
if folder not in child.split(os.pathsep):
    raise SystemExit("ninja directory was not inherited by a child process: " + child)
print("ninja_visible", resolved)
PY
}
log "installing runtime hook into the venv"
install_runtime >"$REPORT/runtime_hook.log" 2>&1

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  DETECTED=$("$PY" -c 'import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"{major}.{minor}")' 2>/dev/null || true)
  if [[ -n "${DETECTED:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="$DETECTED"
    log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST from the visible GPU (no +PTX)"
  else
    log "no visible GPU; leaving TORCH_CUDA_ARCH_LIST unset (GPU builds stay UNTESTED)"
  fi
fi

stage_nvdiffrast() {
  if [[ ! -d "$NV_SRC/.git" ]]; then
    git clone --filter=blob:none https://github.com/ShenhanQian/nvdiffrast.git "$NV_SRC"
  fi
  git -C "$NV_SRC" fetch --filter=blob:none origin "$NV_COMMIT" || true
  git -C "$NV_SRC" checkout --force "$NV_COMMIT"
  git -C "$NV_SRC" clean -fd -e '*.pyc'
  patch -p1 --forward --batch --directory "$NV_SRC" < "$PATCH"
  grep -q 'RenderModeFlag_EnableBackfaceCulling' "$NV_SRC/nvdiffrast/torch/torch_rasterize.cpp"
  grep -q 'glEnable(GL_CULL_FACE)' "$NV_SRC/nvdiffrast/common/rasterize_gl.cpp"
  grep -q 'LAM_COMPAT_PLUGIN_LOAD' "$NV_SRC/nvdiffrast/torch/ops.py"
  if grep -F "os.environ['TORCH_CUDA_ARCH_LIST'] = ''" "$NV_SRC/nvdiffrast/torch/ops.py"; then
    log "nvdiffrast still clears TORCH_CUDA_ARCH_LIST"
    exit 1
  fi
  # Pure-Python package. The .so is JIT-compiled on the first CUDA context.
  timeout --foreground "${NVDIFFRAST_PIP_TIMEOUT:-180}" \
    "$PY" -m pip install --no-build-isolation --no-deps "$NV_SRC" \
    >"$REPORT/nvdiffrast_pip.log" 2>&1
  "$PY" - << 'PY'
import nvdiffrast
from pathlib import Path
ops = (Path(nvdiffrast.__file__).resolve().parent / "torch" / "ops.py").read_text()
loader = ops.split("def _get_plugin", 1)[1].split("def get_log_level", 1)[0]
assert "sys.modules[plugin_name] = module" in loader
assert "module = torch.utils.cpp_extension.load(" in loader
assert "_ensure_ninja_on_subprocess_path()" in loader
assert "except Exception" not in loader
assert "os.environ['TORCH_CUDA_ARCH_LIST'] = ''" not in loader
print("nvdiffrast_package", nvdiffrast.__version__)
PY
}

log "stage nvdiffrast package (not the CUDA JIT)"
stage_nvdiffrast
log "nvdiffrast package install ok; CUDA plugin remains UNTESTED until a real context is created"

log "stage FaceBoxes CPU NMS"
(
  cd "$ROOT/external/landmark_detection/FaceBoxesV2/utils"
  PATH="$(dirname "$PY"):$PATH" \
    timeout --foreground "${FACEBOXES_BUILD_TIMEOUT:-180}" bash make.sh \
    >"$REPORT/faceboxes_build.log" 2>&1
)
"$PY" "$ROOT/scripts/compat/tests/test_faceboxes_cpu_nms.py" --out "$REPORT/faceboxes_cpu_nms.json" \
  >"$REPORT/faceboxes_cpu_nms.log" 2>&1

log "stage PyTorch3D (separate budget; UNTESTED without torch+nvcc)"
set +e
LAM_VENV="$VENV" LAM_REPORT_DIR="$REPORT" LAM_COMPAT_CACHE="$CACHE" \
  bash "$ROOT/scripts/compat/pytorch3d_build.sh" >"$REPORT/pytorch3d_stage.log" 2>&1
PT_RC=$?
set -e
log "pytorch3d_build.sh exit $PT_RC"

"$PY" "$ROOT/scripts/compat/tests/test_plugin_load_mechanism.py" --out "$REPORT/plugin_load_mechanism.json" \
  >"$REPORT/plugin_load_mechanism.log" 2>&1
LAM_NVDIFFRAST_SRC="$NV_SRC" \
  LAM_PYTORCH3D_SRC="${LAM_PYTORCH3D_SRC:-}" \
  "$PY" "$ROOT/scripts/compat/tests/test_static_contracts.py" --out "$REPORT/static_contracts.json" \
  >"$REPORT/static_contracts.log" 2>&1
"$PY" "$ROOT/scripts/compat/tests/run_cuda_kernel_suite.py" --out "$REPORT/cuda_kernel_results.json" \
  >"$REPORT/cuda_kernel_results.log" 2>&1

log "bootstrap finished. GPU kernel results are in $REPORT/cuda_kernel_results.json"
"$PY" - << PY
import json
from pathlib import Path
report = Path("$REPORT")
names = [
    "faceboxes_cpu_nms.json",
    "plugin_load_mechanism.json",
    "static_contracts.json",
    "cuda_kernel_results.json",
    "pytorch3d_build.json",
]
summary = {"report_dir": str(report), "files": {}}
for name in names:
    path = report / name
    summary["files"][name] = json.loads(path.read_text()) if path.exists() else None
(report / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(report / "bootstrap_summary.json")
PY
