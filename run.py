"""Run the Call QA demo: serve the dashboard, or check it from the terminal.

    python demos/call-qa/run.py                              serve the dashboard on http://127.0.0.1:8000
    python demos/call-qa/run.py --smoke                      offline self check, prints SMOKE OK ...
    python demos/call-qa/run.py --check --backend offline    before and after run, exits 1 if the fix does not help

Same as `python -m callqa` from inside demos/call-qa. Every option is listed by --help.
"""
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

from callqa.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
