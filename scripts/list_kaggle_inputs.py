"""List what Kaggle actually mounted, and print the exact paths to paste.

Run this first in every notebook. Kaggle rewrites dataset and notebook-output
names into slugs, so the path you need is rarely the one you expect -- this
prints it ready to copy into --data, --weights, --init-from or --resume.
"""

from pathlib import Path


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f}{unit}" if unit == "B" else f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}GB"


def section(title: str, paths: list[Path], show_size: bool = False) -> None:
    print()
    print(f"{title}:")
    if not paths:
        print("  - none")
        return
    for path in sorted(paths):
        size = f"   [{human(path.stat().st_size)}]" if show_size and path.is_file() else ""
        print(f"  - {path}{size}")


def main() -> None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        print("/kaggle/input does not exist. This script is intended for Kaggle.")
        return

    print("Mounted inputs:")
    for path in sorted(input_root.iterdir()):
        print(f"  - {path}")

    section("Dataset configs (pass to --data)", sorted(input_root.rglob("data.yaml")))

    checkpoints = [p for p in input_root.rglob("*.pt") if p.is_file()]
    section("Checkpoints (pass to --weights / --init-from / --resume)", checkpoints, show_size=True)

    videos: list[Path] = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        videos.extend(input_root.rglob(ext))
    section("Videos (pass to --source)", videos, show_size=True)

    if checkpoints:
        best = next((p for p in checkpoints if p.name == "best.pt"), checkpoints[0])
        print()
        print("Example:")
        print(f"  python scripts/eval_dinov3_detector.py --weights {best} --split test")


if __name__ == "__main__":
    main()
