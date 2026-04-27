#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:09:23 2026

@author: jfcaetano

# GOOGLE COLAB: Compare RandomForest, GradientBoosting, XGBoost
# Target: Label (0/1), where 1 = toxic
# - Selected RDKit descriptors only
# - 64/16/20 train/valid/test split
# - Hyperparameter tuning
# - Validation threshold tuning
# - Final comparison on test set
"""


# 1) Install dependencies

!pip -q install rdkit xgboost tqdm


# 2) Mount Google Drive

from google.colab import drive
drive.mount('/content/drive')


# 3) Imports

import os
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

tqdm.pandas()


# 4) USER SETTINGS

DATA_FILENAME = "/content/drive/MyDrive/ChemEng/acute_dermal.csv"
OUTPUT_DIR = "/content/drive/MyDrive/ChemEng/results_compare_rf_gb_xgb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMILES_COL = "SMILES"
TARGET_COL = "Label"
KEEP_ID_COLS = ["InChIKey"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_SIZE_WITHIN_TRAIN = 0.20
MAX_ABS_VALUE = 1e10

CV_FOLDS = 3
THRESHOLDS = np.arange(0.30, 0.71, 0.02)
MIN_TOXIC_PRECISION = 0.58

# Keep this moderate for speed
N_ITER_RF = 10
N_ITER_GB = 10
N_ITER_XGB = 12


# 5) Descriptor selection

my_descriptors = list()
for desc_name in dir(Descriptors):
    if desc_name in ['BalabanJ', 'BertzCT', 'TPSA']:
        my_descriptors.append(desc_name)
    elif desc_name[:3] == 'Chi':
        my_descriptors.append(desc_name)
    elif 'VSA' in desc_name:
        my_descriptors.append(desc_name)
    elif 'Kappa' in desc_name:
        my_descriptors.append(desc_name)
    elif desc_name[:1] == 'H':
        my_descriptors.append(desc_name)
    elif desc_name[:1] == 'N':
        my_descriptors.append(desc_name)
    elif desc_name[:1] == 'M':
        my_descriptors.append(desc_name)

SELECTED_DESCRIPTORS = []
for desc_name in my_descriptors:
    if hasattr(Descriptors, desc_name):
        obj = getattr(Descriptors, desc_name)
        if callable(obj):
            SELECTED_DESCRIPTORS.append(desc_name)

SELECTED_DESCRIPTORS = sorted(list(set(SELECTED_DESCRIPTORS)))
print(f"Number of selected RDKit descriptors: {len(SELECTED_DESCRIPTORS)}")


# 6) SMILES -> descriptor dict

def smiles_to_descriptor_dict(smiles):
    values = {}

    if pd.isna(smiles) or str(smiles).strip() == "":
        for desc_name in SELECTED_DESCRIPTORS:
            values[desc_name] = np.nan
        return values

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        for desc_name in SELECTED_DESCRIPTORS:
            values[desc_name] = np.nan
        return values

    for desc_name in SELECTED_DESCRIPTORS:
        try:
            func = getattr(Descriptors, desc_name)
            val = func(mol)

            if val is None:
                val = np.nan
            elif isinstance(val, (int, float, np.integer, np.floating)):
                if not np.isfinite(val):
                    val = np.nan
                elif abs(val) > MAX_ABS_VALUE:
                    val = np.nan

            values[desc_name] = val
        except Exception:
            values[desc_name] = np.nan

    return values


# 7) Load data

df = pd.read_csv(DATA_FILENAME)
print("Loaded dataset shape:", df.shape)
print("Columns found:", df.columns.tolist())

required_cols = [SMILES_COL, TARGET_COL]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

existing_keep_cols = [c for c in KEEP_ID_COLS if c in df.columns]
work_df = df[existing_keep_cols + [SMILES_COL, TARGET_COL]].copy()


# 8) Compute descriptors

print(f"\nCalculating selected RDKit descriptors from {SMILES_COL} ...")

descriptor_df = work_df[SMILES_COL].progress_apply(
    lambda x: pd.Series(smiles_to_descriptor_dict(x))
)

full_df = pd.concat([work_df.copy(), descriptor_df], axis=1)

descriptor_csv_path = os.path.join(OUTPUT_DIR, "dataset_with_selected_rdkit_descriptors.csv")
full_df.to_csv(descriptor_csv_path, index=False)
print(f"Saved descriptor dataset to:\n{descriptor_csv_path}")


# 9) Prepare X and y

full_df[TARGET_COL] = pd.to_numeric(full_df[TARGET_COL], errors="coerce")
full_df = full_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
full_df = full_df[full_df[TARGET_COL].isin([0, 1])].reset_index(drop=True)
full_df[TARGET_COL] = full_df[TARGET_COL].astype(int)

X = full_df[SELECTED_DESCRIPTORS].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.mask(X.abs() > MAX_ABS_VALUE, np.nan)

all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    print(f"Dropping {len(all_nan_cols)} all-NaN columns")
    X = X.drop(columns=all_nan_cols)

constant_cols = X.columns[X.nunique(dropna=True) <= 1].tolist()
if constant_cols:
    print(f"Dropping {len(constant_cols)} constant/near-constant columns")
    X = X.drop(columns=constant_cols)

y = full_df[TARGET_COL]

print("\nClass counts:")
print(y.value_counts())

neg_count = int((y == 0).sum())
pos_count = int((y == 1).sum())
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
print(f"\nscale_pos_weight: {scale_pos_weight:.4f}")

cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_selected_rdkit_dataset.csv")
pd.concat([full_df[existing_keep_cols + [SMILES_COL, TARGET_COL]], X], axis=1).to_csv(cleaned_path, index=False)


# 10) Split data

X_train_full, X_test, y_train_full, y_test, idx_train_full, idx_test = train_test_split(
    X, y, full_df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(
    X_train_full, y_train_full, idx_train_full,
    test_size=VALID_SIZE_WITHIN_TRAIN,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print("\nSplit sizes:")
print("Train:", X_train.shape[0])
print("Validation:", X_valid.shape[0])
print("Test:", X_test.shape[0])

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# 11) Search spaces

search_configs = {
    "RandomForest": {
        "n_iter": N_ITER_RF,
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ]),
        "params": {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.5, 0.7],
            "model__bootstrap": [True, False]
        }
    },
    "GradientBoosting": {
        "n_iter": N_ITER_GB,
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ))
        ]),
        "params": {
            "model__n_estimators": [100, 150, 250, 400],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__max_features": ["sqrt", "log2", None, 0.5]
        }
    },
    "XGBoost": {
        "n_iter": N_ITER_XGB,
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                tree_method="hist"
            ))
        ]),
        "params": {
            "model__n_estimators": [150, 250, 400, 600],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.5, 0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 2, 4, 6],
            "model__gamma": [0, 0.1, 0.3, 0.5],
            "model__reg_alpha": [0, 0.01, 0.1, 0.5],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0]
        }
    }
}


# 12) Helper functions

def summarize_results(model_name, y_true, y_pred, y_prob):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob)
    }

def choose_threshold(y_true, y_prob, thresholds, min_precision=0.58):
    rows = []
    best_thr = 0.50
    best_score = -1

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if prec >= min_precision:
            score = 0.45 * rec + 0.45 * f1 + 0.10 * prec
        else:
            score = -1

        rows.append({
            "Threshold": thr,
            "Validation_Accuracy": acc,
            "Validation_Precision": prec,
            "Validation_Recall": rec,
            "Validation_F1": f1,
            "Selection_Score": score
        })

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr, best_score, pd.DataFrame(rows)


# 13) Tune each model

search_summaries = []
candidate_results = []
candidate_artifacts = []

for model_name, config in search_configs.items():
    print(f"\nTuning {model_name} ...")

    search = RandomizedSearchCV(
        estimator=config["pipeline"],
        param_distributions=config["params"],
        n_iter=config["n_iter"],
        scoring="average_precision",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    search_summaries.append({
        "Model": model_name,
        "Best_CV_AveragePrecision": search.best_score_,
        "Best_Params": str(search.best_params_)
    })

    pd.DataFrame(search.cv_results_).to_csv(
        os.path.join(OUTPUT_DIR, f"{model_name}_full_search_results.csv"),
        index=False
    )

    # threshold tuning on validation
    best_model.fit(X_train, y_train)
    valid_prob = best_model.predict_proba(X_valid)[:, 1]
    best_thr, best_score, threshold_df = choose_threshold(
        y_valid, valid_prob, THRESHOLDS, min_precision=MIN_TOXIC_PRECISION
    )

    threshold_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{model_name}_validation_threshold_search.csv"),
        index=False
    )

    # refit on train+validation
    best_model.fit(X_train_full, y_train_full)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred_default = (test_prob >= 0.50).astype(int)
    test_pred_tuned = (test_prob >= best_thr).astype(int)

    result_default = summarize_results(
        f"{model_name}_default_0.50", y_test, test_pred_default, test_prob
    )
    result_default["Chosen_Threshold"] = 0.50

    result_tuned = summarize_results(
        f"{model_name}_tuned_{best_thr:.2f}", y_test, test_pred_tuned, test_prob
    )
    result_tuned["Chosen_Threshold"] = best_thr

    candidate_results.extend([result_default, result_tuned])

    candidate_artifacts.append({
        "base_model_name": model_name,
        "best_threshold": best_thr,
        "model": best_model,
        "test_prob": test_prob,
        "test_pred_default": test_pred_default,
        "test_pred_tuned": test_pred_tuned
    })

# save tuning summary
search_summary_df = pd.DataFrame(search_summaries).sort_values(
    "Best_CV_AveragePrecision", ascending=False
).reset_index(drop=True)

search_summary_path = os.path.join(OUTPUT_DIR, "model_tuning_summary.csv")
search_summary_df.to_csv(search_summary_path, index=False)

print("\nTuning summary:")
print(search_summary_df)


# 14) Rank final candidates

results_df = pd.DataFrame(candidate_results)

results_df["Ranking_Score"] = (
    0.45 * results_df["Recall"] +
    0.35 * results_df["F1"] +
    0.15 * results_df["PR_AUC"] +
    0.05 * results_df["Precision"]
)

results_df = results_df.sort_values(
    ["Ranking_Score", "F1", "PR_AUC", "ROC_AUC"],
    ascending=False
).reset_index(drop=True)

results_path = os.path.join(OUTPUT_DIR, "all_candidate_results_ranked.csv")
results_df.to_csv(results_path, index=False)

print("\nRanked final results:")
print(results_df)

best_variant_name = results_df.iloc[0]["Model"]
print(f"\nBest overall candidate: {best_variant_name}")


# 15) Recover best candidate

best_artifact = None
for artifact in candidate_artifacts:
    base_name = artifact["base_model_name"]
    tuned_name = f"{base_name}_tuned_{artifact['best_threshold']:.2f}"
    default_name = f"{base_name}_default_0.50"

    if best_variant_name == tuned_name or best_variant_name == default_name:
        best_artifact = artifact
        break

if best_artifact is None:
    raise RuntimeError("Could not match best candidate artifact.")

if best_variant_name.endswith("default_0.50"):
    final_pred = best_artifact["test_pred_default"]
else:
    final_pred = best_artifact["test_pred_tuned"]

final_prob = best_artifact["test_prob"]
best_model = best_artifact["model"]


# 16) Save hold-out predictions

test_meta_cols = existing_keep_cols + [SMILES_COL, TARGET_COL]
test_meta = full_df.loc[idx_test, test_meta_cols].reset_index(drop=True)

pred_df = test_meta.copy()
pred_df.rename(columns={TARGET_COL: "y_true"}, inplace=True)
pred_df["y_prob"] = final_prob
pred_df["y_pred"] = final_pred

pred_path = os.path.join(OUTPUT_DIR, "holdout_predictions_best_model.csv")
pred_df.to_csv(pred_path, index=False)


# 17) Final report and confusion matrix

final_report = pd.DataFrame(
    classification_report(y_test, final_pred, output_dict=True, zero_division=0)
).transpose()

final_report_path = os.path.join(OUTPUT_DIR, "best_model_classification_report.csv")
final_report.to_csv(final_report_path)

final_cm = pd.DataFrame(
    confusion_matrix(y_test, final_pred),
    index=["Actual_0", "Actual_1"],
    columns=["Pred_0", "Pred_1"]
)

final_cm_path = os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.csv")
final_cm.to_csv(final_cm_path)

print("\nBest final classification report:")
print(final_report)

print("\nBest final confusion matrix:")
print(final_cm)


# 18) Feature importance if available

model_step = best_model.named_steps["model"]

if hasattr(model_step, "feature_importances_"):
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model_step.feature_importances_
    }).sort_values("importance", ascending=False)

    fi_path = os.path.join(OUTPUT_DIR, "best_model_feature_importance.csv")
    fi.to_csv(fi_path, index=False)

    print("\nTop 20 features:")
    print(fi.head(20))
    print(f"\nSaved feature importance to:\n{fi_path}")

print(f"\nSaved tuning summary to:\n{search_summary_path}")
print(f"Saved ranked candidate results to:\n{results_path}")
print(f"Saved best predictions to:\n{pred_path}")
print(f"Saved best classification report to:\n{final_report_path}")
print(f"Saved best confusion matrix to:\n{final_cm_path}")
print(f"\nThe End")
