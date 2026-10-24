"""Step 1 CLI:  python run_pipeline.py data/videos/demo.mp4 [--no-yolo] [--force]"""
import argparse
import json
import sys
from pathlib import Path

from core import pipeline


def main():
    ap = argparse.ArgumentParser(description="VisionGuard detection + tracking pipeline")
    ap.add_argument("video", type=Path)
    ap.add_argument("--no-yolo", action="store_true", help="use the motion fallback detector")
    ap.add_argument("--force", action="store_true", help="ignore cache and reprocess")
    args = ap.parse_args()

    def progress(p, msg):
        print(f"\r  {p*100:5.1f}%  {msg}      ", end="", flush=True)

    try:
        meta = pipeline.run(args.video, prefer_yolo=not args.no_yolo, force=args.force,
                            on_progress=progress)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nError: {e}")
        sys.exit(1)
    print()
    if meta.get("warning"):
        print("WARNING:", meta["warning"])
    print(json.dumps(meta, indent=2))
    print(f"\nResults saved in {pipeline.cache_dir_for(args.video)}")


if __name__ == "__main__":
    main()
