"""Fetch the GolfPose source revision used by this pipeline (weights separate)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "third_party/GolfPose"
REVISION = "0b749c6df4bb7f060ab42fa3dbd71de91bde17c1"


def main():
    if TARGET.exists():
        current = subprocess.check_output(["git", "-C", str(TARGET), "rev-parse", "HEAD"], text=True).strip()
        if current != REVISION:
            raise RuntimeError(f"Existing GolfPose is {current}; expected {REVISION}. Keep local work and use a separate checkout.")
        print(f"GolfPose already at {REVISION}")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", "https://github.com/MingHanLee/GolfPose.git", str(TARGET)], check=True)
    subprocess.run(["git", "-C", str(TARGET), "checkout", "--detach", REVISION], check=True)


if __name__ == "__main__":
    main()
