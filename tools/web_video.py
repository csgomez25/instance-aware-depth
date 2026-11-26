"""Re-encode result videos to small H.264 files that play in browsers, GitHub and PowerPoint.

    python web_video.py                       # every mp4 under results/novel_view and results/tracking, in place
    python web_video.py some.mp4 --crf 30     # specific files; higher crf = smaller and blurrier

OpenCV writes mp4v, which is large and does not play on GitHub. Files already in H.264 are skipped.
"""
import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg

import common as C

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def is_h264(path):
    info = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True).stderr
    return "Video: h264" in info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--crf", type=int, default=28)
    args = ap.parse_args()
    videos = [Path(v) for v in args.videos] or sorted(
        p for d in ("novel_view", "tracking") for p in (C.RESULTS / d).glob("*.mp4"))
    for v in videos:
        if is_h264(v):
            print(f"{v.name}: already H.264")
            continue
        tmp = v.with_suffix(".tmp.mp4")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(v), "-c:v", "libx264", "-crf", str(args.crf),
                        "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)], check=True)
        before = v.stat().st_size
        tmp.replace(v)
        print(f"{v.name}: {before / 1e6:.1f} -> {v.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
