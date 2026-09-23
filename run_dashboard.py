import socket
import subprocess
import sys
from pathlib import Path


def find_available_port(candidates=range(8501, 8511), host="127.0.0.1"):
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
                return probe.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free dashboard port found between 8501 and 8510.")


def main():
    app_dir = Path(__file__).resolve().parent
    port = find_available_port()
    print(f"Opening Expiry AI at http://127.0.0.1:{port}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_dir / "app.py"),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(port),
        ],
        cwd=app_dir,
        check=False,
    )


if __name__ == "__main__":
    main()
