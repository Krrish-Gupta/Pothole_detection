import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO

# ── Palette tuned to match the dashboard's dark theme ──────────────────────────
COLOURS = {
    "Low":    "#3ECF8E",
    "Medium": "#F5A623",
    "High":   "#F04747",
    "None":   "#6B7280",
    "Ignore": "#6B7280",
}

TEXT_COLOUR = "#E6E8EB"     # light text, readable on dark background
GRID_COLOUR = "#2A2F3A"
ACCENT      = "#5B8DEF"


def _style_axes(ax):
    """Apply consistent dark-theme styling to any matplotlib axes."""
    ax.set_facecolor("none")
    ax.tick_params(colors=TEXT_COLOUR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOUR)
    ax.yaxis.label.set_color(TEXT_COLOUR)
    ax.title.set_color(TEXT_COLOUR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOUR)


def _fig_to_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.patch.set_alpha(0)
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def severity_donut(summary: dict) -> bytes:
    """Donut chart of Low / Medium / High / None counts."""
    labels  = [k for k in summary if summary[k] > 0]
    values  = [summary[k] for k in labels]
    colours = [COLOURS.get(k, "#888780") for k in labels]

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, _ = ax.pie(
        values, colors=colours,
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "#0E1117", "linewidth": 2}
    )
    total = sum(values)
    ax.text(0, 0, f"{total}\nimages", ha="center", va="center",
            fontsize=14, fontweight="600", color=TEXT_COLOUR)
    legend = [mpatches.Patch(color=COLOURS.get(l, "#888"), label=f"{l}  ·  {summary[l]}")
              for l in labels]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=9, frameon=False, labelcolor=TEXT_COLOUR)
    ax.set_title("Severity distribution", fontsize=12, pad=12, fontweight="600")
    _style_axes(ax)
    return _fig_to_bytes(fig)


def daily_trend(df: pd.DataFrame) -> bytes:
    """Line chart: total defects detected per day."""
    fig, ax = plt.subplots(figsize=(6, 3))
    _style_axes(ax)

    if df.empty:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_COLOUR, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        return _fig_to_bytes(fig)

    ax.plot(df["date"], df["total_defects"],
            color=ACCENT, linewidth=2.5, marker="o", markersize=6,
            markerfacecolor=ACCENT, markeredgecolor="#0E1117", markeredgewidth=1.5)
    ax.fill_between(df["date"], df["total_defects"], alpha=0.15, color=ACCENT)
    ax.set_title("Defects detected per day", fontsize=12, fontweight="600")
    ax.set_ylabel("Defect count")
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.6, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return _fig_to_bytes(fig)


def repair_priority_bar(summary: dict) -> bytes:
    """Bar chart coloured by severity, annotated with percentages."""
    order   = ["High", "Medium", "Low", "None", "Ignore"]
    labels  = [s for s in order if s in summary]
    values  = [summary[s] for s in labels]
    colours = [COLOURS[s] for s in labels]
    total   = sum(values) or 1

    fig, ax = plt.subplots(figsize=(5, 3))
    _style_axes(ax)
    bars = ax.bar(labels, values, color=colours, width=0.55, edgecolor="none")
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values) * 0.02,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=10,
                color=TEXT_COLOUR, fontweight="600")
    ax.set_title("Repair priority breakdown", fontsize=12, fontweight="600")
    ax.set_ylabel("Images")
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.6, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _fig_to_bytes(fig)


def confidence_histogram(df: pd.DataFrame) -> bytes:
    """Histogram of model confidence scores across all detections."""
    fig, ax = plt.subplots(figsize=(5, 3))
    _style_axes(ax)

    data = df[df["confidence"] > 0]["confidence"].dropna()
    if data.empty:
        ax.text(0.5, 0.5, "No detections yet", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_COLOUR, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        return _fig_to_bytes(fig)

    ax.hist(data, bins=20, color=ACCENT, edgecolor="#0E1117", alpha=0.9, linewidth=0.5)
    ax.axvline(data.median(), color="#F5A623", linewidth=2,
               linestyle="--", label=f"Median {data.median():.2f}")
    ax.set_title("Detection confidence distribution", fontsize=12, fontweight="600")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9, frameon=False, labelcolor=TEXT_COLOUR)
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.6, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _fig_to_bytes(fig)
