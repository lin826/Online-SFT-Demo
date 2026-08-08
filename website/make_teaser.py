#!/usr/bin/env python3
"""Draw the article teaser: one request, three routes, one real outcome.

The teaser deliberately reuses the palette and typography of the result
figures so the top of the article does not look like stock artwork.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "figures" / "blog_teaser.png"

INK = "#1f2328"
MUTED = "#6b7280"
FAINT = "#9aa1ab"
LINE = "#dfe2e7"
ACCENT = "#1f5fd8"
ACCENT_SOFT = "#eef3fd"
TEAL = "#0f6f63"
TEAL_SOFT = "#eaf3f1"

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
# Sizes below are authored against the layout grid; this scales them so the
# rendered labels land near 11-12px at the article's figure width.
TEXT_SCALE = 1.1


def _box(axis, x, y, width, height, *, fill="white", edge=LINE, lw=1.0, radius=1.4):
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            zorder=3,
        )
    )


def _arrow(axis, start, end, *, color=LINE, lw=1.2, style="-", connection="arc3,rad=0"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            color=color,
            linewidth=lw,
            linestyle=style,
            connectionstyle=connection,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def _text(axis, x, y, label, *, size=8.5, color=INK, weight="normal", ha="center"):
    axis.text(
        x, y, label,
        ha=ha, va="center", fontsize=size * TEXT_SCALE, color=color,
        fontweight=weight, zorder=4,
    )


def draw_teaser(output_path: Path) -> None:
    matplotlib.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": FONT_STACK}
    )
    figure, axis = plt.subplots(figsize=(7.3, 3.0))
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 40)
    axis.axis("off")

    # Incoming stream of requests.
    for offset, shade in ((3.0, "#f1f3f6"), (1.5, "#e9ecf1"), (0.0, "white")):
        _box(axis, 1.5 + offset, 21 + offset * 0.9, 14.5, 7.4, fill=shade, edge=LINE)
    _text(axis, 8.7, 26.4, "10:47 p.m. receipt", size=7.6, color=INK)
    _text(axis, 8.7, 23.6, "importance 0.31", size=7, color=MUTED)
    _text(axis, 9.0, 17.6, "one request at a time", size=7.4, color=MUTED)

    _arrow(axis, (17.2, 24.7), (22.4, 24.7), color=FAINT, lw=1.1)

    # On-device student.
    _box(axis, 23, 12.6, 22.2, 21.9, fill="white", edge=LINE, lw=1.1)
    _text(axis, 34.1, 31.6, "on-device student", size=8, color=INK, weight="bold")
    _text(axis, 34.1, 28.8, "frozen 230M LFM + LoRA", size=7, color=MUTED)
    routes = (("INTERRUPT", 0.18), ("LATER", 0.29), ("ARCHIVE", 0.53))
    for index, (name, probability) in enumerate(routes):
        y = 25.2 - index * 3.8
        chosen = name == "ARCHIVE"
        _text(axis, 25.4, y, name, size=7, color=INK if chosen else MUTED, ha="left")
        axis.add_patch(
            FancyBboxPatch(
                (34.0, y - 0.55), 7.6, 1.1,
                boxstyle="round,pad=0,rounding_size=0.55",
                facecolor="#eef0f3", edgecolor="none", zorder=3,
            )
        )
        axis.add_patch(
            FancyBboxPatch(
                (34.0, y - 0.55), 7.6 * probability, 1.1,
                boxstyle="round,pad=0,rounding_size=0.55",
                facecolor=ACCENT if chosen else "#b9c1cc", edgecolor="none", zorder=4,
            )
        )
    _text(axis, 34.1, 15.0, "acts before any feedback exists", size=6.6, color=MUTED)

    # Three routes; only one of them is ever executed.
    _text(axis, 67, 37.8, "one route executes", size=7.8, color=INK)
    _text(axis, 67, 35.6, "the other two never produce evidence", size=7, color=MUTED)
    lanes = (("INTERRUPT", 31.0, False), ("LATER", 24.0, False), ("ARCHIVE", 17.0, True))
    for name, y, executed in lanes:
        rad = 0.0 if abs(y - 24.0) < 0.1 else (0.1 if y > 24.0 else -0.1)
        _arrow(
            axis, (45.8, 24.0), (55.4, y),
            color=ACCENT if executed else "#d2d7de",
            lw=1.6 if executed else 1.0,
            style="-" if executed else (0, (3, 2.4)),
            connection=f"arc3,rad={rad}",
        )
        if executed:
            _box(axis, 56, y - 2.9, 22, 5.8, fill=ACCENT_SOFT, edge=ACCENT, lw=1.1)
            _text(axis, 67, y + 1.0, "ARCHIVE executed", size=7.6, color=ACCENT)
            _text(axis, 67, y - 1.3, "NO_OBSERVATION", size=7.2, color=INK)
        else:
            _box(axis, 56, y - 2.5, 22, 5.0, fill="#fafbfc", edge=LINE)
            _text(axis, 67, y + 0.8, name, size=7.2, color=FAINT)
            _text(axis, 67, y - 1.4, "no outcome exists", size=7, color=FAINT)

    # Post-decision teacher.
    _arrow(axis, (78.6, 17.0), (82.4, 17.0), color=ACCENT, lw=1.4)
    _box(axis, 83, 9.5, 15, 17, fill=TEAL_SOFT, edge=TEAL, lw=1.1)
    _text(axis, 90.5, 23.8, "teacher", size=8, color=TEAL, weight="bold")
    for index, (label, value) in enumerate((("I", 0.09), ("L", 0.29), ("A", 0.62))):
        y = 19.6 - index * 2.6
        _text(axis, 84.9, y, label, size=6.8, color=MUTED)
        axis.add_patch(
            FancyBboxPatch(
                (86.4, y - 0.5), 9.8, 1.0,
                boxstyle="round,pad=0,rounding_size=0.5",
                facecolor="white", edgecolor="none", zorder=4,
            )
        )
        axis.add_patch(
            FancyBboxPatch(
                (86.4, y - 0.5), 9.8 * value, 1.0,
                boxstyle="round,pad=0,rounding_size=0.5",
                facecolor=TEAL, edgecolor="none", zorder=5,
            )
        )
    _text(axis, 90.5, 11.5, "soft target over routes", size=6.8, color=TEAL)

    # Update loop back into the student, routed below the row so it cannot be
    # confused with the forward path. The caption sits on the line.
    axis.plot(
        [90.5, 90.5, 34.1, 34.1],
        [9.3, 4.6, 4.6, 10.2],
        color=TEAL, lw=1.3, solid_capstyle="round", solid_joinstyle="round",
        zorder=2,
    )
    _arrow(axis, (34.1, 10.0), (34.1, 12.4), color=TEAL, lw=1.3)
    axis.text(
        62, 4.6,
        "the update helps the next decision, never this one",
        ha="center", va="center", fontsize=7.4 * TEXT_SCALE, color=TEAL, zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 3},
    )

    figure.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    draw_teaser(args.output.resolve())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
