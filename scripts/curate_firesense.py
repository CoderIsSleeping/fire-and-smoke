"""Curate the FIRESENSE videos down to the clips this project can actually use.

FIRESENSE ships 49 videos. Most of them are forest fires, bonfires, fireplaces
and outdoor smoke-bomb tests, which are not what a fixed camera in an
industrial building sees. Rather than evaluating on all of them and reporting a
number that describes the wrong problem, this selects the subset that matches
the deployment scenario:

  positive  fire or smoke in or on a building, viewed from a fixed camera
  negative  fixed-camera scenes containing things that look like fire but are
            not -- night street lighting, headlight glare, warm bokeh, indoor
            lamps, motorway traffic, cloud

The negatives are the more valuable half. Fire footage is easy to find; footage
of a fixed camera watching something fire-like that is not fire, which is what
the low-false-alarm requirement is measured against, is not.

One clip is deliberately excluded from the metrics: testpos05 is the single
most on-target video in the whole dataset (a warehouse interior at night on a
timestamped CCTV camera) but the dataset's own annotation contours are burned
into the pixels. Evaluating on it would measure the model's reaction to cyan
lines as much as to smoke. It is kept separately for qualitative use.

    python scripts/curate_firesense.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "raw" / "firesense"
DST = ROOT / "datasets" / "raw" / "firesense_curated"

# (relative source path, why it was kept)
POSITIVE = [
    ("smoke/pos/testpos02.818.avi", "NIST structure-fire test: smoke venting from a building window, fixed camera"),
    ("fire/pos/posVideo1.868.avi", "flame and smoke along a long industrial building facade"),
    ("fire/pos/posVideo2.871.avi", "indoor kitchen fire - enclosed interior space"),
    ("fire/pos/posVideo6.875.avi", "small ground fire watched by a fixed perimeter camera"),
    ("fire/pos/posVideo7.876.avi", "ground fire, fixed perimeter camera, wider view"),
]

NEGATIVE = [
    ("smoke/neg/testneg03.809.avi", "indoor workshop/office, fixed camera, bright desk lamps"),
    ("smoke/neg/testneg06.812.avi", "motorway traffic, fixed camera"),
    ("smoke/neg/testneg07.813.avi", "cloud moving behind bare trees - classic smoke false positive"),
    ("smoke/neg/testneg09.815.avi", "motorway traffic, fixed camera, different angle"),
    ("fire/neg/negsVideo2.859.avi", "night street: headlights and sodium lighting - classic flame false positive"),
    ("fire/neg/negsVideo3.860.avi", "night street traffic with warm lighting"),
    ("fire/neg/negsVideo7.864.avi", "warm out-of-focus bokeh lights - strong flame-colour false positive"),
    ("fire/neg/negsVideo9.866.avi", "warm bokeh close-up, moving highlights"),
    ("fire/neg/negsVideo10.1072.avi", "office interior, fixed camera, person moving"),
    ("fire/neg/negsVideo12.1074.avi", "motorway overpass, high resolution, fixed camera"),
    ("fire/neg/negsVideo13.1075.avi", "station concourse interior, crowd, fixed camera"),
    ("fire/neg/negsVideo14.1076.avi", "covered station platform, fixed camera"),
]

EXCLUDED = [
    ("smoke/pos/testpos05.821.avi",
     "Warehouse interior at night, timestamped CCTV - the single most on-target clip in "
     "FIRESENSE. EXCLUDED from metrics because the dataset's annotation contours are "
     "burned into the pixels, so any score would partly measure the model's reaction to "
     "those overlays. Use it for qualitative demonstration only."),
]


def copy_group(items: list[tuple[str, str]], target: Path) -> list[tuple[str, str, bool]]:
    target.mkdir(parents=True, exist_ok=True)
    results = []
    for rel, reason in items:
        src = SRC / rel
        if not src.exists():
            print(f"  MISSING {rel}")
            results.append((rel, reason, False))
            continue
        shutil.copy2(src, target / src.name)
        results.append((rel, reason, True))
    return results


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Download and extract FIRESENSE first.")

    if DST.exists():
        shutil.rmtree(DST)

    print("curating FIRESENSE ->", DST)
    pos = copy_group(POSITIVE, DST / "positive")
    neg = copy_group(NEGATIVE, DST / "negative")
    exc = copy_group(EXCLUDED, DST / "excluded_annotated")

    lines = [
        "# FIRESENSE - curated subset",
        "",
        "Selected by `scripts/curate_firesense.py` from the 49-video FIRESENSE database",
        "(Zenodo record 836749, CC-BY 4.0).",
        "",
        "FIRESENSE is **not** an industrial fire dataset. Most of it is forest fire,",
        "bonfire, fireplace and outdoor smoke-bomb footage. This subset keeps only the",
        "clips that resemble a fixed camera watching a building or an interior, because",
        "evaluating on the rest would produce a number about the wrong problem.",
        "",
        "The negatives matter more than the positives here. Fire footage is easy to find;",
        "fixed-camera footage of fire-*like* scenes that contain no fire is what the",
        "low-false-alarm requirement is actually measured against.",
        "",
        f"## positive/ ({sum(1 for _, _, ok in pos if ok)} clips)",
        "",
    ]
    for rel, reason, ok in pos:
        if ok:
            lines.append(f"- `{Path(rel).name}` - {reason}")
    lines += ["", f"## negative/ ({sum(1 for _, _, ok in neg if ok)} clips)", ""]
    for rel, reason, ok in neg:
        if ok:
            lines.append(f"- `{Path(rel).name}` - {reason}")
    lines += ["", "## excluded_annotated/ (not used for metrics)", ""]
    for rel, reason, ok in exc:
        if ok:
            lines.append(f"- `{Path(rel).name}` - {reason}")
    lines += [
        "",
        "## Running the evaluation",
        "",
        "```bash",
        "python scripts/eval_videos_dinov3.py \\",
        "    --weights <stage2 best.pt> \\",
        "    --videos datasets/raw/firesense_curated \\",
        "    --stride 3",
        "```",
        "",
        "`excluded_annotated/` is neither a positive nor a negative folder name, so those",
        "clips are listed as unlabelled and left out of the summary automatically.",
        "",
        "## Honest limitation",
        "",
        "None of this is a factory. It validates that the temporal confirmation layer",
        "behaves on real fixed-camera video, and it gives a first video-level false-alarm",
        "rate. It does not validate performance on the deployment site. Real site footage",
        "is still required.",
        "",
    ]
    (DST / "CURATION.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"  positive : {sum(1 for _, _, ok in pos if ok)}")
    print(f"  negative : {sum(1 for _, _, ok in neg if ok)}")
    print(f"  excluded : {sum(1 for _, _, ok in exc if ok)}")
    print(f"  wrote {DST / 'CURATION.md'}")


if __name__ == "__main__":
    main()
