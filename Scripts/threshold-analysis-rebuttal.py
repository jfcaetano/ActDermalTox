#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 00:33:29 2026

@author: jfcaetano
"""

import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "rdkit": "rdkit",
    "xgboost": "xgboost",
    "shap": "shap",
    "tqdm": "tqdm",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "imblearn": "imbalanced-learn",
}

for module_name, package_name in REQUIRED_PACKAGES.items():
    if importlib.util.find_spec(module_name) is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package_name]
        )



from google.colab import drive

drive.mount("/content/drive")



import os
import io
import contextlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from IPython.display import display, Image



OUTPUT_DIR = (
    "/content/drive/MyDrive/ChemEng/"
    "results_scaffold_locked_qsar_vif_smote_adasyn"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTION_FILE = os.path.join(
    OUTPUT_DIR,
    "locked_scaffold_test_predictions_with_ad.csv",
)

CSV_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "locked_test_threshold_sensitivity_056_vs_034.csv",
)

IMAGE_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "locked_test_threshold_sensitivity_056_vs_034.png",
)



PRIMARY_THRESHOLD = 0.56
ALTERNATIVE_THRESHOLD = 0.34


df = pd.read_csv(PREDICTION_FILE)

y_true = df["y_true"].astype(int).to_numpy()
y_prob = df["y_probability_toxic"].astype(float).to_numpy()


def evaluate_threshold(y_true, y_prob, threshold):

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    return {
        "Threshold": threshold,
        "TP": int(tp),
        "FN": int(fn),
        "TN": int(tn),
        "FP": int(fp),
        "Precision_toxic": precision,
        "Recall_toxic": recall,
        "Specificity_non_toxic": specificity,
        "F1_toxic": f1,
        "Accuracy": accuracy,
        "BalancedAccuracy": balanced_accuracy,
        "MCC": mcc,
        "Toxic_stopped_percent": 100 * recall,
        "Non_toxic_allowed_percent": 100 * specificity,
    }


results = pd.DataFrame(
    [
        evaluate_threshold(
            y_true,
            y_prob,
            PRIMARY_THRESHOLD,
        ),
        evaluate_threshold(
            y_true,
            y_prob,
            ALTERNATIVE_THRESHOLD,
        ),
    ]
)



pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", True)


pd.set_option("display.width", 110)

pd.set_option("display.precision", 6)

primary = results.iloc[0]
alternative = results.iloc[1]


log_buffer = io.StringIO()

with contextlib.redirect_stdout(log_buffer):

    print("LOCKED-TEST THRESHOLD SENSITIVITY ANALYSIS")
    print()

    print(results.round(6))

    print()
    print("PRACTICAL EFFECT OF LOWERING THE THRESHOLD")
    print()

    print(
        f"False negatives: "
        f"{int(primary['FN'])} -> "
        f"{int(alternative['FN'])}"
    )

    print(
        f"False positives: "
        f"{int(primary['FP'])} -> "
        f"{int(alternative['FP'])}"
    )

    print(
        f"Toxic recall: "
        f"{primary['Recall_toxic']:.3f} -> "
        f"{alternative['Recall_toxic']:.3f}"
    )

    print(
        f"Toxic precision: "
        f"{primary['Precision_toxic']:.3f} -> "
        f"{alternative['Precision_toxic']:.3f}"
    )

    print(
        f"Specificity: "
        f"{primary['Specificity_non_toxic']:.3f} -> "
        f"{alternative['Specificity_non_toxic']:.3f}"
    )

    print()
    print("Saved comparison to:")
    print(CSV_OUTPUT_FILE)


log_text = log_buffer.getvalue()


print(log_text)
results.to_csv(
    CSV_OUTPUT_FILE,
    index=False,
)


lines = log_text.splitlines()

number_of_lines = len(lines)
max_line_length = max(len(line) for line in lines)

figure_width = max(10, max_line_length * 0.075)
figure_height = max(5, number_of_lines * 0.25)

fig = plt.figure(
    figsize=(figure_width, figure_height),
    facecolor="white",
)

plt.axis("off")

fig.text(
    0.01,
    0.99,
    log_text,
    ha="left",
    va="top",
    family="monospace",
    fontsize=9,
    linespacing=1.25,
)

plt.savefig(
    IMAGE_OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.25,
    facecolor="white",
)

plt.close(fig)



print("\nFILES SAVED")
print("CSV:")
print(CSV_OUTPUT_FILE)

print("\nPNG:")
print(IMAGE_OUTPUT_FILE)

display(
    Image(
        filename=IMAGE_OUTPUT_FILE
    )
)
