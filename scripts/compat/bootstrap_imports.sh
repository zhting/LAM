#!/usr/bin/env bash
# Weightless import-stage installer. Reuses the kernel venv and its torch.
# Does not build PyTorch3D, nvdiffrast, GS, or xformers from source.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CACHE=${LAM_COMPAT_CACHE:-"$HOME/.cache/lam-blackwell"}
REPORT=${LAM_REPORT_DIR:-"$CACHE/report"}
VENV=${LAM_VENV:-"$ROOT/.venv-blackwell"}
PY=${LAM_PYTHON:-"$VENV/bin/python"}
TIMEOUT=${IMPORT_STAGE_TIMEOUT:-600}
mkdir -p "$REPORT"
if [[ ! -x "$PY" ]]; then
  echo "FAIL_EARLY: missing interpreter $PY. Run scripts/compat/bootstrap_blackwell.sh first." >&2
  exit 2
fi

BEFORE=$("$PY" -c 'import torch; print(torch.__version__)')
"$PY" "$ROOT/scripts/compat/import_plan.py" --write "$REPORT/import_plan.json"
PLAN=$("$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["pair"]["torchvision"]); print(p["pair"]["xformers"]); print(p["pair"]["index"]); print(p["accelerate"]); print(p["diffusers"])' "$REPORT/import_plan.json")
mapfile -t LINES <<< "$PLAN"
TV=${LINES[0]}
XF=${LINES[1]}
INDEX=${LINES[2]}
ACCEL=${LINES[3]}
DIFFUSERS=${LINES[4]}

"$PY" - << PY > "$REPORT/import_constraints.txt"
import torch, numpy
print(f"torch=={torch.__version__}")
print(f"numpy=={numpy.__version__}")
PY

echo "import stage torch=$BEFORE torchvision=$TV xformers=$XF cap=${TIMEOUT}s"
"$PY" "$ROOT/scripts/compat/preflight_imports.py" --report "$REPORT/import_preflight.json"

# Pure wheels first. The constraint blocks a silent torch or numpy replacement.
timeout --foreground "$TIMEOUT" "$PY" -m pip install --upgrade-strategy only-if-needed \
  -c "$REPORT/import_constraints.txt" \
  -r "$ROOT/scripts/compat/constraints-import-cu130.txt" \
  >"$REPORT/import_pure.log" 2>&1

# Binary ABI wheels. --no-deps so their torch requirement cannot replace the runtime torch.
timeout --foreground 180 "$PY" -m pip install --no-deps --index-url "$INDEX" \
  "torchvision==$TV" \
  >"$REPORT/import_torchvision.log" 2>&1
set +e
timeout --foreground 180 "$PY" -m pip install --no-deps --index-url "$INDEX" \
  "xformers==$XF" \
  >"$REPORT/import_xformers.log" 2>&1
XF_RC=$?
set -e
if [[ "$XF_RC" -ne 0 ]]; then
  echo "xformers wheel install failed (rc=$XF_RC). Not building from source. See $REPORT/import_xformers.log" \
    | tee "$REPORT/import_xformers_status.txt"
else
  echo "xformers_wheel_installed" | tee "$REPORT/import_xformers_status.txt"
fi

timeout --foreground 120 "$PY" -m pip install --no-deps \
  -c "$REPORT/import_constraints.txt" \
  "accelerate==$ACCEL" "diffusers==$DIFFUSERS" \
  >"$REPORT/import_nodeps.log" 2>&1

# No cp313 wheel. Bound the sdist compile; failure does not roll back the other packages.
set +e
timeout --foreground "${PYMCUBES_TIMEOUT:-120}" "$PY" -m pip install --no-deps "pymcubes==0.1.6" \
  >"$REPORT/import_pymcubes.log" 2>&1
MC_RC=$?
set -e
echo "pymcubes_rc=$MC_RC" | tee "$REPORT/import_pymcubes_status.txt"

if ! compgen -G "$ROOT/external/landmark_detection/FaceBoxesV2/utils/nms/cpu_nms*.so" >/dev/null; then
  echo "FaceBoxes extension missing; rebuilding CPU NMS (previously measured about 3s)"
  (
    cd "$ROOT/external/landmark_detection/FaceBoxesV2/utils"
    PATH="$(dirname "$PY"):$PATH" timeout --foreground "${FACEBOXES_BUILD_TIMEOUT:-180}" bash make.sh
  ) >"$REPORT/import_faceboxes_build.log" 2>&1
fi

AFTER=$("$PY" -c 'import torch; print(torch.__version__)')
if [[ "$AFTER" != "$BEFORE" ]]; then
  echo "FAIL_EARLY: torch changed from $BEFORE to $AFTER" >&2
  exit 3
fi
set +e
"$PY" -m pip check >"$REPORT/import_pip_check.log" 2>&1
CHECK_RC=$?
set -e
echo "pip_check_rc=$CHECK_RC" | tee "$REPORT/import_pip_check_status.txt"
if [[ "$CHECK_RC" -ne 0 ]]; then
  echo "FAIL_EARLY: pip check reported unsatisfied requirements. See $REPORT/import_pip_check.log" >&2
  exit 4
fi
echo "torch_unchanged $AFTER"
echo "import bootstrap done"
