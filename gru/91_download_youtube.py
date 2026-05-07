import argparse
import sys
from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func


def parse_time(value: str) -> float:
    parts = value.strip().split(":")
    if not all(p for p in parts):
        raise ValueError(f"Invalid time: {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def download(
    url: str,
    output_dir: Path,
    fps: int = 30,
    start: float | None = None,
    end: float | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fps_filter = f"[fps<={fps}]"
    format_selector = (
        f"bestvideo{fps_filter}[ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo{fps_filter}+bestaudio/"
        f"best{fps_filter}"
    )

    downloaded: list[str] = []

    def hook(d: dict) -> None:
        if d.get("status") == "finished":
            downloaded.append(d["info_dict"].get("filepath") or d["filename"])

    ydl_opts = {
        "format": format_selector,
        "merge_output_format": "mp4",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ],
        "progress_hooks": [hook],
        "noplaylist": True,
    }

    if cookies_from_browser:
        browser, _, profile = cookies_from_browser.partition(":")
        ydl_opts["cookiesfrombrowser"] = (browser, profile or None, None, None)
    if cookies_file is not None:
        ydl_opts["cookiefile"] = str(cookies_file)

    if start is not None or end is not None:
        s = 0.0 if start is None else start
        e = float("inf") if end is None else end
        if e <= s:
            raise ValueError(f"end ({e}) must be greater than start ({s})")
        ydl_opts["download_ranges"] = download_range_func(None, [(s, e)])
        ydl_opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    final = Path(downloaded[-1]) if downloaded else output_dir
    return final.with_suffix(".mp4")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a YouTube video as mp4 capped at a target fps.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path.cwd(), help="Destination folder (default: cwd)")
    parser.add_argument("--fps", type=int, default=30, help="Maximum fps to accept (default: 30)")
    parser.add_argument("--start", type=parse_time, default=None, help="Start time, e.g. 90, 1:30, or 0:01:30")
    parser.add_argument("--end", type=parse_time, default=None, help="End time, same format as --start")
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Load cookies from a logged-in browser, e.g. 'chrome', 'firefox', 'edge', or 'chrome:Profile 1'",
    )
    parser.add_argument("--cookies", type=Path, default=None, help="Path to a Netscape-format cookies.txt file")
    args = parser.parse_args()

    path = download(
        args.url,
        args.output_dir,
        args.fps,
        args.start,
        args.end,
        args.cookies_from_browser,
        args.cookies,
    )
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
