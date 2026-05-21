"""Run the lightweight smoke suite and optional README GIF checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str]) -> int:
    print("$ " + " ".join(args), flush=True)
    return subprocess.call(args, cwd=ROOT)


def check_gifs() -> int:
    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as exc:
        print(f"GIF check requires the viz extra: {exc}", file=sys.stderr)
        return 1

    gif_dir = ROOT / "docs" / "assets" / "gifs"
    paths = sorted(gif_dir.glob("*.gif"))
    if not paths:
        print(f"no GIFs found in {gif_dir}", file=sys.stderr)
        return 1

    failed = False
    for path in paths:
        frames = imageio.mimread(path)
        if not frames:
            print(f"empty GIF: {path.relative_to(ROOT)}", file=sys.stderr)
            failed = True
            continue
        first = np.asarray(frames[0])
        last = np.asarray(frames[-1])
        first_std = float(first.std())
        last_std = float(last.std())
        print(
            f"{path.relative_to(ROOT)}: "
            f"frames={len(frames)} first_std={first_std:.2f} last_std={last_std:.2f}"
        )
        if len(frames) < 2 or max(first_std, last_std) < 2.0:
            failed = True

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gifs", action="store_true", help="regenerate README GIFs")
    parser.add_argument(
        "--check-gifs",
        action="store_true",
        help="check GIF frame counts and nonblank pixels",
    )
    args = parser.parse_args()

    status = run_command([sys.executable, "-m", "pytest", "-q"])
    if status != 0:
        return status

    if args.gifs:
        status = run_command([sys.executable, "scripts/make_gifs.py"])
        if status != 0:
            return status

    if args.check_gifs:
        return check_gifs()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
