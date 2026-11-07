#!/usr/bin/env python3
"""Reproduce the torch 2.11 extension-load gap with a real native module.

This is not a CUDA test and must not be reported as GPU validation.
torch 2.11 torch/utils/cpp_extension.py:_import_module_from_library loads a
shared object with spec_from_file_location and returns it without inserting
sys.modules. nvdiffrast then called importlib.import_module(name) and raised
ModuleNotFoundError: nvdiffrast_plugin.
"""

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

SOURCE = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *plugin_ping(PyObject *self, PyObject *args) {
    return PyLong_FromLong(7);
}

static PyMethodDef methods[] = {
    {"ping", plugin_ping, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "synthetic_plugin",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_synthetic_plugin(void) {
    return PyModule_Create(&moduledef);
}
"""


def load_like_torch_211(module_name, filepath):
    """Same control flow as PyTorch 2.11.0 _import_module_from_library."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create a spec for {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish_like_patch(module, module_name):
    if module is None:
        return importlib.import_module(module_name)
    sys.modules[module_name] = module
    if not hasattr(module, "ping"):
        raise RuntimeError("compiled plugin is missing the expected symbol")
    return module


def compile_plugin(directory, module_name, include):
    source = directory / f"{module_name}.c"
    binary = directory / f"{module_name}.so"
    source.write_text(SOURCE.replace("synthetic_plugin", module_name))
    subprocess.check_call(
        ["gcc", "-shared", "-fPIC", "-I", include, "-o", str(binary), str(source)],
        stdout=subprocess.DEVNULL,
    )
    return binary


def assert_missing(module_name):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise SystemExit(f"unexpected missing name {exc.name!r}")
        return
    raise SystemExit(f"import of {module_name} succeeded; expected ModuleNotFoundError")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    include = sysconfig.get_path("include")
    with tempfile.TemporaryDirectory(prefix="lam-plugin-") as tmp:
        directory = Path(tmp)
        binary = compile_plugin(directory, "synthetic_plugin", include)
        for name in ("synthetic_plugin", "synthetic_plugin_v1"):
            sys.modules.pop(name, None)
        loaded = load_like_torch_211("synthetic_plugin", binary)
        if loaded.ping() != 7:
            raise SystemExit("file-location load did not return a working module")
        # The G4 failure is import-by-bare-name after load() returned. A
        # same-named extension may be inserted as a side effect of
        # create_dynamic; dropping that entry recreates the observed state
        # without pretending the CUDA plugin was executed.
        sys.modules.pop("synthetic_plugin", None)
        assert_missing("synthetic_plugin")
        if loaded.ping() != 7:
            raise SystemExit("the returned module stopped working once it was not importable by name")
        publish_like_patch(loaded, "synthetic_plugin")
        if importlib.import_module("synthetic_plugin").ping() != 7:
            raise SystemExit("publishing the returned module did not restore import")

        # torch 2.11 _jit_compile renames a rebuilt extension to name_vN and
        # returns that module. import_module(unversioned) cannot see it.
        versioned = compile_plugin(directory, "synthetic_plugin_v1", include)
        loaded_v1 = load_like_torch_211("synthetic_plugin_v1", versioned)
        sys.modules.pop("synthetic_plugin", None)
        assert_missing("synthetic_plugin")
        if loaded_v1.ping() != 7:
            raise SystemExit("versioned module object did not execute")
        publish_like_patch(loaded_v1, "synthetic_plugin")
        if importlib.import_module("synthetic_plugin").ping() != 7:
            raise SystemExit("aliasing the versioned module under the unversioned name failed")

    result = {
        "test": "native module returned by file-location load, then published",
        "status": "PASS",
        "kind": "NATIVE_MODULE_LOAD_MECHANISM",
        "gpu": "UNTESTED",
        "torch_reference": "v2.11.0 torch/utils/cpp_extension.py _jit_compile returns the loaded module and renames rebuilds to name_vN",
        "reproduced_exception": "ModuleNotFoundError: synthetic_plugin",
        "not_a_gpu_result": True,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
