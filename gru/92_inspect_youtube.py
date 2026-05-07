import argparse
import sys
from pathlib import Path

import yt_dlp


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "?"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def inspect(
    url: str,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
) -> int:
    ydl_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }
    if cookies_from_browser:
        browser, _, profile = cookies_from_browser.partition(":")
        ydl_opts["cookiesfrombrowser"] = (browser, profile or None, None, None)
    if cookies_file is not None:
        ydl_opts["cookiefile"] = str(cookies_file)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    print(f"Title:    {info.get('title', '?')}")
    print(f"Uploader: {info.get('uploader', '?')}")
    print(f"Duration: {format_duration(info.get('duration'))}")

    formats = info.get("formats") or []
    video_formats = [f for f in formats if f.get("vcodec") and f["vcodec"] != "none"]

    best_per_resolution: dict[tuple[int, float], dict] = {}
    for f in video_formats:
        fps = f.get("fps")
        height = f.get("height")
        if fps is None or height is None:
            continue
        key = (int(height), round(float(fps), 3))
        current = best_per_resolution.get(key)
        if current is None or (f.get("tbr") or 0) > (current.get("tbr") or 0):
            best_per_resolution[key] = f

    if not best_per_resolution:
        print("\nNo video streams with reported fps were found.")
        return 1

    print("\nAvailable video streams (best per resolution+fps):")
    print(f"  {'Height':>6}  {'FPS':>7}  {'Codec':<14}  {'Ext':<5}  Note")
    for (height, fps), f in sorted(best_per_resolution.items(), reverse=True):
        codec = (f.get("vcodec") or "?").split(".")[0]
        ext = f.get("ext") or "?"
        note = f.get("format_note") or ""
        print(f"  {height:>6}  {fps:>7.3f}  {codec:<14}  {ext:<5}  {note}")

    fps_values = sorted({fps for _, fps in best_per_resolution}, reverse=True)
    print(f"\nFrame rates offered: {', '.join(f'{v:g}' for v in fps_values)} fps")
    print(f"Source frame rate (best guess): {fps_values[0]:.3f} fps")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a YouTube video's metadata and frame rates without downloading."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Load cookies from a logged-in browser, e.g. 'firefox', 'chrome', 'edge', or 'chrome:Profile 1'",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Path to a Netscape-format cookies.txt file",
    )
    args = parser.parse_args()
    return inspect(args.url, args.cookies_from_browser, args.cookies)


if __name__ == "__main__":
    sys.exit(main())
