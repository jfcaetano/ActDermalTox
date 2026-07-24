#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 00:10:46 2026

@author: jfcaetano
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from IPython.display import Image, display


# ============================================================
# 0. Configure
# ============================================================

try:
    fm.findfont("Arial", fallback_to_default=False)
    figure_font = "Arial"
    print("Arial font found and selected.")

except ValueError:
    # Arial may not be installed in some Google Colab environments
    figure_font = "Liberation Sans"
    print(
        "Arial was not found. Liberation Sans will be used as a fallback.\n"
        "Install or upload Arial to Colab and rerun the code for exact Arial."
    )

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [figure_font],
        "font.size": 14,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,

        # Preserve editable TrueType text in exported files
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# 1. Locate the threshold-performance CSV file
# ============================================================

search_root = Path("/content/drive/MyDrive/ChemEng")

matches = list(
    search_root.rglob("development_threshold_search.csv")
)

if not matches:
    raise FileNotFoundError(
        "development_threshold_search.csv was not found "
        "inside /content/drive/MyDrive/ChemEng."
    )

# Use the most recently modified matching file
csv_path = max(
    matches,
    key=lambda path: path.stat().st_mtime,
)

output_dir = csv_path.parent

print(f"Reading data from: {csv_path}")
print(f"Saving figures to: {output_dir}")


# ============================================================
# 2. Load and validate the threshold data
# ============================================================

threshold_table = pd.read_csv(csv_path)

# Remove accidental spaces from column names
threshold_table.columns = (
    threshold_table.columns
    .astype(str)
    .str.strip()
)

required_columns = [
    "Threshold",
    "Precision_toxic",
    "Recall_toxic",
    "F1_toxic",
]

missing_columns = set(required_columns).difference(
    threshold_table.columns
)

if missing_columns:
    raise ValueError(
        f"Required columns are missing: "
        f"{sorted(missing_columns)}\n"
        f"Available columns: "
        f"{threshold_table.columns.tolist()}"
    )

# Convert required columns to numeric values
for column in required_columns:
    threshold_table[column] = pd.to_numeric(
        threshold_table[column],
        errors="coerce",
    )

threshold_table = threshold_table.dropna(
    subset=required_columns
)

threshold_table = threshold_table.sort_values(
    by="Threshold"
).reset_index(drop=True)

if threshold_table.empty:
    raise ValueError(
        "The threshold table contains no valid numerical data."
    )


# ============================================================
# 3. Select the operating threshold
# ============================================================

minimum_precision = 0.58

eligible = threshold_table[
    threshold_table["Precision_toxic"] >= minimum_precision
].copy()

if eligible.empty:
    raise ValueError(
        "No classification threshold satisfies "
        f"toxic-class precision ≥ {minimum_precision:.2f}."
    )

# Maximize toxic-class F1-score.
# Use toxic-class recall as the secondary criterion.
selected_row = eligible.sort_values(
    by=[
        "F1_toxic",
        "Recall_toxic",
    ],
    ascending=[
        False,
        False,
    ],
).iloc[0]

selected_threshold = float(
    selected_row["Threshold"]
)

selected_precision = float(
    selected_row["Precision_toxic"]
)

selected_recall = float(
    selected_row["Recall_toxic"]
)

selected_f1 = float(
    selected_row["F1_toxic"]
)

print("\nSelected operating point")
print("------------------------")
print(f"Threshold: {selected_threshold:.2f}")
print(f"Precision: {selected_precision:.3f}")
print(f"Recall:    {selected_recall:.3f}")
print(f"F1-score:  {selected_f1:.3f}")


# ============================================================
# 4. Create the threshold-performance figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 6.0)
)

ax.plot(
    threshold_table["Threshold"],
    threshold_table["Precision_toxic"],
    marker="o",
    markersize=5,
    linewidth=2,
    label="Toxic-class precision",
)

ax.plot(
    threshold_table["Threshold"],
    threshold_table["Recall_toxic"],
    marker="s",
    markersize=5,
    linewidth=2,
    label="Toxic-class recall",
)

ax.plot(
    threshold_table["Threshold"],
    threshold_table["F1_toxic"],
    marker="^",
    markersize=5,
    linewidth=2,
    label="Toxic-class F1-score",
)

# Minimum precision requirement
ax.axhline(
    y=minimum_precision,
    linestyle=":",
    linewidth=2,
    label=f"Minimum precision = {minimum_precision:.2f}",
)

# Selected classification threshold
ax.axvline(
    x=selected_threshold,
    linestyle="--",
    linewidth=2,
    label=f"Selected threshold = {selected_threshold:.2f}",
)

# Highlight the selected F1-score
ax.scatter(
    selected_threshold,
    selected_f1,
    s=100,
    facecolors="none",
    edgecolors="black",
    linewidths=2,
    zorder=5,
)

# Annotate selected operating point
ax.annotate(
    (
        f"Threshold = {selected_threshold:.2f}\n"
        f"Precision = {selected_precision:.3f}\n"
        f"Recall = {selected_recall:.3f}\n"
        f"F1 = {selected_f1:.3f}"
    ),
    xy=(
        selected_threshold,
        selected_f1,
    ),
    xytext=(12, 30),
    textcoords="offset points",
    fontsize=18,
    fontfamily=figure_font,
    ha="left",
    va="bottom",
)


# ============================================================
# 5. Format the figure
# ============================================================

ax.set_xlabel(
    "Classification threshold",
    fontsize=17,
    fontfamily=figure_font,
)

ax.set_ylabel(
    "Metric value",
    fontsize=20,
    fontfamily=figure_font,
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=20,
)

# Explicitly apply the font to tick labels
for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
    tick_label.set_fontfamily(figure_font)

ax.set_xlim(
    threshold_table["Threshold"].min(),
    threshold_table["Threshold"].max(),
)

ax.set_ylim(0, 1)

ax.grid(
    alpha=0.25,
)

# Legend without surrounding box
legend = ax.legend(
    fontsize=17,
    frameon=False,
    loc="best",
)

# Explicitly apply Arial to legend text
for legend_text in legend.get_texts():
    legend_text.set_fontfamily(figure_font)

fig.tight_layout()


# ============================================================
# 6. Save PNG and PDF versions
# ============================================================

png_path = (
    output_dir
    / "development_threshold_performance_curve_Arial.png"
)

pdf_path = (
    output_dir
    / "development_threshold_performance_curve_Arial.pdf"
)

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# 7. Confirm that the files were created
# ============================================================

print("\nSaved figure files")
print("------------------")

for path in [png_path, pdf_path]:
    if path.exists():
        print(
            f"{path.name}: created successfully "
            f"({path.stat().st_size:,} bytes)"
        )
    else:
        print(
            f"{path.name}: file was not created"
        )

# Display the saved PNG inside Colab
if png_path.exists():
    display(
        Image(filename=str(png_path))
    )
