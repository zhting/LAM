#!/usr/bin/env python3
"""Offline checks for the import plan. No torch and no GPU."""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "compat"))

import import_plan


def main() -> None:
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    pair = import_plan.pair_for("2.14.1+cu130")
    add("torch_2_14_1_pair", pair["torchvision"] == "0.29.1+cu130" and pair["xformers"] == "0.0.35", str(pair))
    try:
        import_plan.pair_for("2.3.0+cu121")
        add("unknown_torch_fails", False, "did not exit")
    except SystemExit as exc:
        add("unknown_torch_fails", "FAIL_EARLY" in str(exc), str(exc))

    pins = import_plan.pure_pins()
    text = (ROOT / "scripts" / "compat" / "constraints-import-cu130.txt").read_text()
    add("constraints_match_plan", pins == [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")], str(len(pins)))
    add("no_numpy_downgrade_pin", not any(pin.startswith("numpy==1.") for pin in pins), "numpy")
    add("moviepy_stays_on_1", "moviepy==1.0.3" in pins, "moviepy.editor")

    smoke = (ROOT / "scripts" / "compat" / "tests" / "smoke_full_imports.py").read_text()
    tree = ast.parse(smoke)
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"ModelLAM", "FlameHead", "FlameTrackingSingleImage", "GlobalTracker", "FaceBoxesDetector", "VGGHeadDetector", "StyleMatteEngine", "Dinov2FusionWrapper"}:
                forbidden.append(node.func.id)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    add(
        "smoke_does_not_construct_asset_loaders",
        not forbidden and "load_state_dict_from_url" not in calls and "load" not in calls,
        str(forbidden or calls),
    )
    add("smoke_marks_blocked_resource", "BLOCKED_RESOURCE" in smoke and "construct_flame_head" in smoke, "blocked list")
    bootstrap = (ROOT / "scripts" / "compat" / "bootstrap_imports.sh").read_text()
    add("bootstrap_no_deps_for_abi_wheels", "--no-deps" in bootstrap and "torchvision==$TV" in bootstrap, "bootstrap")
    add("bootstrap_refuses_torch_change", 'torch changed' in bootstrap, "version compare")
    attention = (ROOT / "lam/models/encoders/dinov2/layers/attention.py").read_text()
    add(
        "upstream_xformers_fallback_unchanged",
        "except ImportError" in attention and "XFORMERS_AVAILABLE = False" in attention and "return super().forward(x)" in attention,
        "dinov2 attention",
    )

    failed = [item["name"] for item in checks if item["status"] == "FAIL"]
    document = {"test": "import plan", "status": "FAIL" if failed else "PASS", "checks": checks, "cuda": "UNTESTED"}
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    rendered = json.dumps(document, indent=2)
    print(rendered)
    if out:
        out.write_text(rendered + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
