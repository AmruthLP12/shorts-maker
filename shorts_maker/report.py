from pathlib import Path


def _fmt(seconds: float) -> str:
    """Human-readable duration: 1m 23s or 45s."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def write_markdown_report(
    path: Path,
    index: int,
    source_label: str,
    metadata: dict,
) -> None:
    """
    Write a Markdown file summarising all publishing metadata for one Short.

    Parameters
    ----------
    path          : destination .md file
    index         : short number (1-based)
    source_label  : YouTube URL or absolute path of the source video
    metadata      : the short_metadata dict written to the .json file
    """

    clip = metadata["clip"]
    content = metadata["content"]
    yt = metadata.get("youtube", {})
    ig = metadata.get("instagram", {})

    start = clip["start"]
    end = clip["end"]
    duration = clip["duration"]
    score = clip["score"]

    title = content["title"]
    hook = content.get("hook", "")
    reason = content.get("reason", "")
    description = content.get("description", "")
    tags = content.get("tags", [])
    hashtags = content.get("hashtags", [])

    yt_title = yt.get("title", title)
    yt_desc = yt.get("description", description)
    yt_tags = yt.get("tags", tags)
    yt_hashtags = yt.get("hashtags", hashtags)

    ig_caption = ig.get("caption", description)
    ig_hashtags = ig.get("hashtags", hashtags)

    # Score bar: filled/empty squares
    filled = "█" * score
    empty = "░" * (10 - score)
    score_bar = f"{filled}{empty} {score}/10"

    lines = [
        f"# Short {index:02d} — {title}",
        "",
        f"> *{hook}*",
        "",
        f"| Field    | Value |",
        f"|----------|-------|",
        f"| **Score** | `{score_bar}` |",
        f"| **Clip**  | `{_fmt(start)} → {_fmt(end)}` ({_fmt(duration)}) |",
        f"| **Source** | `{source_label}` |",
        f"| **Files** | `short-{index:02d}.mp4` · `short-{index:02d}.srt` · `short-{index:02d}.json` · `short-{index:02d}-thumbnail.png` |",
        "",
        "---",
        "",
        "## Why this works as a Short",
        "",
        reason or "_No reason provided._",
        "",
        "---",
        "",
        "## Description",
        "",
        description or "_No description provided._",
        "",
        "---",
        "",
        "## Tags & Hashtags",
        "",
        "**Tags:** " + " · ".join(f"`{t}`" for t in tags),
        "",
        "**Hashtags:** " + "  ".join(hashtags),
        "",
        "---",
        "",
        "## YouTube",
        "",
        f"**Title:** {yt_title}",
        "",
        "**Description:**",
        "",
        yt_desc or "_No description provided._",
        "",
        "**Tags:** " + " · ".join(f"`{t}`" for t in yt_tags),
        "",
        "**Hashtags:** " + "  ".join(yt_hashtags),
        "",
        "---",
        "",
        "## Instagram",
        "",
        "**Caption:**",
        "",
        ig_caption or "_No caption provided._",
        "",
        "**Hashtags:** " + "  ".join(ig_hashtags),
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
