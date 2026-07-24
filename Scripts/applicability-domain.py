#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 00:33:29 2026

@author: jfcaetano
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# 1. Locate the locked-test prediction file

search_root = Path("/content/drive/MyDrive/ChemEng")

matches = list(
    search_root.rglob(
        "locked_scaffold_test_predictions_with_ad.csv"
    )
)

if not matches:
    raise FileNotFoundError(
        "locked_scaffold_test_predictions_with_ad.csv "
        "was not found."
    )

prediction_path = max(
    matches,
    key=lambda path: path.stat().st_mtime,
)

output_dir = prediction_path.parent

print(f"Reading: {prediction_path}")
print(f"Saving results to: {output_dir}")


# ============================================================
# 2. Load and validate the data
# ============================================================

data = pd.read_csv(prediction_path)
data.columns = data.columns.astype(str).str.strip()

required_columns = {
    "y_true",
    "y_probability_toxic",
    "inside_applicability_domain",
}

missing_columns = required_columns - set(data.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}\n"
        f"Available columns: {data.columns.tolist()}"
    )

data["y_true"] = pd.to_numeric(
    data["y_true"],
    errors="raise",
).astype(int)

data["y_probability_toxic"] = pd.to_numeric(
    data["y_probability_toxic"],
    errors="coerce",
)

# Use the fixed development-selected threshold
selected_threshold = 0.56

if "y_prediction" in data.columns:
    data["y_prediction"] = pd.to_numeric(
        data["y_prediction"],
        errors="coerce",
    ).astype(int)
else:
    data["y_prediction"] = (
        data["y_probability_toxic"]
        >= selected_threshold
    ).astype(int)


# ============================================================
# 3. Normalize applicability-domain labels
# ============================================================

def normalize_ad_value(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    inside_values = {
        "true",
        "1",
        "yes",
        "inside",
        "in",
        "within",
    }

    outside_values = {
        "false",
        "0",
        "no",
        "outside",
        "out",
    }

    if text in inside_values:
        return True

    if text in outside_values:
        return False

    return np.nan


data["inside_ad_normalized"] = data[
    "inside_applicability_domain"
].apply(normalize_ad_value)

if data["inside_ad_normalized"].isna().any():
    invalid_values = data.loc[
        data["inside_ad_normalized"].isna(),
        "inside_applicability_domain",
    ].unique()

    raise ValueError(
        "Some applicability-domain values could not be "
        f"interpreted: {invalid_values}"
    )


# ============================================================
# 4. Metric calculation function
# ============================================================

def calculate_metrics(group_name, subset):
    y_true = subset["y_true"].to_numpy()
    y_pred = subset["y_prediction"].to_numpy()
    y_prob = subset[
        "y_probability_toxic"
    ].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    toxic_count = int((y_true == 1).sum())
    nontoxic_count = int((y_true == 0).sum())

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    # ROC-AUC and PR-AUC require both classes
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )

        pr_auc = average_precision_score(
            y_true,
            y_prob,
        )
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "Applicability_domain": group_name,
        "Total_N": len(subset),
        "Nontoxic_N": nontoxic_count,
        "Toxic_N": toxic_count,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "Toxic_precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Toxic_recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Toxic_F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Specificity": specificity,
        "MCC": matthews_corrcoef(
            y_true,
            y_pred,
        ),
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Brier_score": brier_score_loss(
            y_true,
            y_prob,
        ),
    }


# ============================================================
# 5. Calculate overall, inside-AD, and outside-AD metrics
# ============================================================

inside_data = data[
    data["inside_ad_normalized"] == True
].copy()

outside_data = data[
    data["inside_ad_normalized"] == False
].copy()

results = pd.DataFrame(
    [
        calculate_metrics(
            "All locked-test compounds",
            data,
        ),
        calculate_metrics(
            "Inside applicability domain",
            inside_data,
        ),
        calculate_metrics(
            "Outside applicability domain",
            outside_data,
        ),
    ]
)

numeric_columns = results.select_dtypes(
    include=[np.number]
).columns

results[numeric_columns] = results[
    numeric_columns
].round(3)


# ============================================================
# 6. Save and display the results
# ============================================================

csv_output = (
    output_dir
    / "applicability_domain_performance_metrics.csv"
)

txt_output = (
    output_dir
    / "applicability_domain_performance_metrics.txt"
)

results.to_csv(
    csv_output,
    index=False,
)

with txt_output.open(
    "w",
    encoding="utf-8",
) as log_file:
    log_file.write(
        "APPLICABILITY-DOMAIN PERFORMANCE ANALYSIS\n"
    )
    log_file.write("=" * 70 + "\n")
    log_file.write(
        f"Prediction file: {prediction_path}\n"
    )
    log_file.write(
        f"Fixed threshold: {selected_threshold:.2f}\n\n"
    )
    log_file.write(
        results.to_string(index=False)
    )
    log_file.write("\n")

print("\nApplicability-domain performance")
print("--------------------------------")
print(results.to_string(index=False))

print("\nSaved files")
print("-----------")
print(csv_output)
print(txt_output)
