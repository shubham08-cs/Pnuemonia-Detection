import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    app_path = root / "app" / "app.py"

    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    env["STREAMLIT_SERVER_PORT"] = "8501"

    python_executable = sys.executable
    args = [python_executable, "-m", "streamlit", "run", str(app_path)]

    return subprocess.call(args, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
