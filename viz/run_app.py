"""
run_app.py -- launches both the backend (FastAPI/uvicorn, port 8000) and
frontend (Vite dev server, port 5173) for the BTCUSDT HMM Regime Viewer,
and opens the app in the default browser.

Assumes scripts/build_regime_chart_data.py has already been run at least
once (Data/chart/*.parquet exist) and `npm install` has been run inside
viz/frontend/. See viz/README.md for first-time setup.

Run: `python viz/run_app.py` (from the project root, or from viz/ -- both
resolve paths relative to this file, not the current working directory).
"""

import os
import subprocess
import sys
import time
import webbrowser

_VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_VIZ_DIR)
BACKEND_DIR = os.path.join(_VIZ_DIR, "backend")
FRONTEND_DIR = os.path.join(_VIZ_DIR, "frontend")
CHART_DATA_DIR = os.path.join(_ROOT, "Data", "chart")


def main():
    if not os.path.exists(os.path.join(CHART_DATA_DIR, "models.json")):
        print("ERROR: Data/chart/models.json not found.")
        print("Run this first:  python scripts/build_regime_chart_data.py")
        sys.exit(1)

    if not os.path.exists(os.path.join(FRONTEND_DIR, "node_modules")):
        print("ERROR: frontend dependencies not installed.")
        print(f"Run this first:  cd {FRONTEND_DIR} && npm install")
        sys.exit(1)

    print("Starting backend (FastAPI) on http://127.0.0.1:8000 ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"],
        cwd=BACKEND_DIR,
    )

    print("Starting frontend (Vite) on http://localhost:5173 ...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_proc = subprocess.Popen([npm_cmd, "run", "dev"], cwd=FRONTEND_DIR)

    time.sleep(3)
    print("\nOpening http://localhost:5173 in your browser...")
    try:
        webbrowser.open("http://localhost:5173")
    except Exception:
        pass

    print("\nBoth servers running. Press Ctrl+C to stop.")
    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping servers...")
        backend_proc.terminate()
        frontend_proc.terminate()


if __name__ == "__main__":
    main()
