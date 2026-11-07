"""Make venv build tools visible when Python is started by absolute path.

torch 2.14 is_ninja_available() runs ['ninja', '--version'] and inherits PATH.
An isolated venv's ninja lives in <venv>/bin/ninja. Launching
<venv>/bin/python without activating the venv does not put that directory
on PATH, so the import fails even though the wheel is installed.

This module publishes that absolute executable on os.environ['PATH'] before
any child (ninja, nvcc, the host compiler) is spawned. It also rewrites a
C++14/17 standard flag to C++20 on compiler argv that this process spawns.
Ninja writes its own compiler command lines, so PyTorch3D's setup.py must
also request C++20; this rewriter covers direct spawns such as the distutils
fallback observed on G4.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig

_ACTIVATED = False
_WRAPPED = False
LAST_COMPILER_COMMANDS: list[list[str]] = []

_STD_UPGRADE = {
    "-std=c++14": "-std=c++20",
    "-std=c++17": "-std=c++20",
    "-std=gnu++14": "-std=gnu++20",
    "-std=gnu++17": "-std=gnu++20",
    "/std:c++14": "/std:c++20",
    "/std:c++17": "/std:c++20",
}


def _ninja_names() -> list[str]:
    if os.name == "nt":
        return ["ninja.exe", "ninja"]
    return ["ninja"]


def _candidate_paths() -> list[str]:
    candidates: list[str] = []
    pinned = os.environ.get("LAM_NINJA", "").strip()
    if pinned:
        candidates.append(pinned)
    directories = [os.path.dirname(os.path.realpath(sys.executable))]
    scripts = sysconfig.get_path("scripts")
    if scripts:
        directories.append(scripts)
    try:
        import ninja as ninja_dist
    except ImportError:
        ninja_dist = None
    if ninja_dist is not None:
        bin_dir = getattr(ninja_dist, "BIN_DIR", None)
        if bin_dir:
            directories.append(str(bin_dir))
    for directory in directories:
        for name in _ninja_names():
            candidates.append(os.path.join(directory, name))
    which = shutil.which("ninja")
    if which:
        candidates.append(which)
    unique: list[str] = []
    for path in candidates:
        if path and path not in unique:
            unique.append(path)
    return unique


def _runnable(path: str) -> bool:
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return False
    try:
        subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def require_ninja() -> str:
    """Return an absolute ninja path and publish it for child processes."""
    tried = _candidate_paths()
    for path in tried:
        if _runnable(path):
            folder = os.path.dirname(os.path.abspath(path))
            current = os.environ.get("PATH", "")
            parts = [item for item in current.split(os.pathsep) if item]
            if folder not in parts:
                os.environ["PATH"] = folder + (os.pathsep + current if current else "")
            resolved = os.path.abspath(path)
            os.environ["LAM_NINJA"] = resolved
            return resolved
    raise RuntimeError(
        "Ninja is required, but no executable could be run. "
        "Python executable: %s. Searched: %s. PATH: %s"
        % (sys.executable, tried, os.environ.get("PATH", ""))
    )


def _is_compiler(executable: str) -> bool:
    base = os.path.basename(executable).lower()
    if base in {"nvcc", "nvcc.exe", "c++", "g++", "gcc", "clang", "clang++", "cc"}:
        return True
    stems = ("nvcc", "g++", "gcc", "clang++", "clang", "c++")
    return any(base == stem or base.startswith(stem + "-") or ("-" + stem) in base for stem in stems)


def _upgrade_flag(flag: str) -> str:
    replacement = _STD_UPGRADE.get(flag)
    if replacement:
        return replacement
    updated = flag
    for old, new in _STD_UPGRADE.items():
        if old in updated:
            updated = updated.replace(old, new)
    return updated


def rewrite_compiler_command(args):
    """Upgrade C++14/17 flags on host-compiler and nvcc argument lists.

    Strings and non-compiler commands are returned unchanged. A failing
    compiler still fails; only the standard flag is changed.
    """
    if isinstance(args, tuple):
        args = list(args)
    if not isinstance(args, list) or not args or not isinstance(args[0], str):
        return args
    if not _is_compiler(args[0]):
        return args
    rewritten = [_upgrade_flag(arg) if isinstance(arg, str) else arg for arg in args]
    collapsed: list = []
    saw_std = False
    for arg in rewritten:
        if isinstance(arg, str) and (arg.startswith("-std=") or arg.startswith("/std:")):
            if saw_std:
                continue
            saw_std = True
        collapsed.append(arg)
    if collapsed != list(args):
        LAST_COMPILER_COMMANDS.append([str(item) for item in collapsed])
        del LAST_COMPILER_COMMANDS[:-20]
    return collapsed


def _install_subprocess_wrapper() -> None:
    global _WRAPPED
    if _WRAPPED:
        return
    original = subprocess.Popen

    class _RewritingPopen(original):
        def __init__(self, args, *pargs, **kwargs):
            super().__init__(rewrite_compiler_command(args), *pargs, **kwargs)

    subprocess.Popen = _RewritingPopen
    _WRAPPED = True


def activate() -> str | None:
    """Idempotent startup hook. Returns the ninja path, or None if absent."""
    global _ACTIVATED
    _install_subprocess_wrapper()
    if _ACTIVATED:
        return os.environ.get("LAM_NINJA")
    _ACTIVATED = True
    try:
        return require_ninja()
    except RuntimeError:
        return None


activate()
