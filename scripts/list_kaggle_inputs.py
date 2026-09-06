from pathlib import Path


def main() -> None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        print("/kaggle/input does not exist. This script is intended for Kaggle.")
        return

    print("Kaggle inputs:")
    for path in sorted(input_root.iterdir()):
        print(f"- {path}")

    print("\nDetected detection dataset config files:")
    yamls = sorted(input_root.rglob("data.yaml"))
    if not yamls:
        print("- none")
    for path in yamls:
        print(f"- {path}")

    print("\nDetected videos:")
    videos = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        videos.extend(input_root.rglob(ext))
    if not videos:
        print("- none")
    for path in sorted(videos):
        print(f"- {path}")


if __name__ == "__main__":
    main()
