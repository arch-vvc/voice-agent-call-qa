"""Shared test setup: make `callqa` and `evals` importable from any directory.

The demo folder has a hyphen in its name, so it is not a package. We put it on sys.path instead, at the
end. Every demo has a run.py, and the exception triage tests do `import run`: if this folder came first
on sys.path in a whole repo test run, they would get this demo's run.py by mistake.
"""
import sys
from pathlib import Path

DEMO_DIR = str(Path(__file__).resolve().parents[1])
sys.path[:] = [p for p in sys.path if p != DEMO_DIR]  # pytest may already have put it first
sys.path.append(DEMO_DIR)
