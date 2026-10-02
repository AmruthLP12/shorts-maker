from pathlib import Path
import yt_dlp


def download_video(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / "source.mp4"

    if output.exists():
        print(f"[download] using cached video: {output}")
        return output

    ydl_opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "outtmpl": str(output_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output
