"""Fail when a lock file no longer covers its requirements file: a package
added to requirements.txt without regenerating the lock, or a version range
changed so the pinned version falls outside it.

    python tools/check_locks.py

Regenerate the locks with the commands in README.md (Dependencies).
"""
import re
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ((("requirements.txt",), "requirements.lock"),
         (("requirements.txt", "requirements-dev.txt"),
          "requirements-dev.lock"),
         (("infra/requirements.txt",), "infra/requirements.lock"))
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^ ;\\]+)")


def _wanted(path):
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield Requirement(line)


def _pinned(path):
    out = {}
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        m = PIN.match(line)
        if m:
            out.setdefault(canonicalize_name(m.group(1)), []).append(
                m.group(2))
    return out


def problems():
    found = []
    for sources, lock in PAIRS:
        pinned = _pinned(lock)
        for src in sources:
            for req in _wanted(src):
                versions = pinned.get(canonicalize_name(req.name))
                if not versions:
                    found.append(f"{lock}: {req.name} (from {src}) is not "
                                 f"pinned")
                    continue
                if not any(req.specifier.contains(v, prereleases=True)
                           for v in versions):
                    found.append(f"{lock}: {req.name}=={versions[0]} is "
                                 f"outside {req.specifier} (from {src})")
    return found


if __name__ == "__main__":
    bad = problems()
    for line in bad:
        print(line)
    if bad:
        sys.exit("The locks are out of date: regenerate them (README.md, "
                 "Dependencies).")
    print("Locks cover their requirements.")
