#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:33:17 2026

@author: jfcaetano

# GOOGLE COLAB CODE - XGBoost toxicity model with GO / NO-GO evaluation
# FEATURE & SHAP ANALYSIS
# Target: Label (0/1), where 1 = toxic
#
# Decision interpretation:
# - GO = predicted non-toxic (0)
# - NO-GO = predicted toxic (1)
#
"""


# 1) Install dependencies
!pip -q install rdkit xgboost tqdm shap

# 2) Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3) Imports

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

tqdm.pandas()

# 4) USER SETTINGS
DATA_FILENAME = "/content/drive/MyDrive/ChemEng/acute_dermal.csv"
OUTPUT_DIR = "/content/drive/MyDrive/ChemEng/results_xgboost_shap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMILES_COL = "SMILES"
TARGET_COL = "Label"
KEEP_ID_COLS = ["InChIKey"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_SIZE_WITHIN_TRAIN = 0.20
MAX_ABS_VALUE = 1e10
TOP_K = 60
MAX_SHAP_DISPLAY = 20

BEST_PARAMS = {
    "subsample": 0.7,
    "reg_lambda": 10.0,
    "reg_alpha": 0.5,
    "n_estimators": 500,
    "min_child_weight": 1,
    "max_depth": 6,
    "learning_rate": 0.05,
    "gamma": 0.3,
    "colsample_bytree": 0.8
}

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

required_cols = [SMILES_COL, TARGET_COL]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

existing_keep_cols = [c for c in KEEP_ID_COLS if c in df.columns]
work_df = df[existing_keep_cols + [SMILES_COL, TARGET_COL]].copy()

# 8) Compute descriptors
print(f"\nCalculating selected RDKit descriptors from {SMILES_COL} ...")

descriptor_df = work_df[SMILES_COL].progress_apply(
    lambda x: pd.Series(smiles_to_descriptor_dict(x)))

full_df = pd.concat([work_df.copy(), descriptor_df], axis=1)

descriptor_csv_path = os.path.join(OUTPUT_DIR, "dataset_with_selected_rdkit_descriptors.csv")
full_df.to_csv(descriptor_csv_path, index=False)
print(f"Saved descriptor dataset to:\n{descriptor_csv_path}")

# 9) Prepare Features
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
    X = X.drop(columns=all_nan_cols)

constant_cols = X.columns[X.nunique(dropna=True) <= 1].tolist()
if constant_cols:
    X = X.drop(columns=constant_cols)

y = full_df[TARGET_COL]

print("\nClass counts:")
print(y.value_counts())

neg_count = int((y == 0).sum())
pos_count = int((y == 1).sum())
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
print(f"\nscale_pos_weight: {scale_pos_weight:.4f}")

# 10) Split data
X_train_full, X_test, y_train_full, y_test, idx_train_full, idx_test = train_test_split(
    X, y, full_df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y)

X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(
    X_train_full, y_train_full, idx_train_full,
    test_size=VALID_SIZE_WITHIN_TRAIN,
    random_state=RANDOM_STATE,
    stratify=y_train_full)

print("\nSplit sizes:")
print("Train:", X_train.shape[0])
print("Validation:", X_valid.shape[0])
print("Test:", X_test.shape[0])

# 11) Helper function
def make_xgb(params=None, random_state=42):
    base = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist")
    if params is not None:
        base.update(params)
    return XGBClassifier(**base)

# 12) Train model for top-60 feature selection
imputer_fs = SimpleImputer(strategy="median")
X_train_imp_fs = imputer_fs.fit_transform(X_train)
X_valid_imp_fs = imputer_fs.transform(X_valid)

feature_model = make_xgb(BEST_PARAMS, random_state=RANDOM_STATE)
feature_model.fit(X_train_imp_fs, y_train, verbose=False)

feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": feature_model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

feature_importance_df["Percentage"] = (
    feature_importance_df["importance"] / feature_importance_df["importance"].sum()
) * 100

feature_importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance_for_shap_selection.csv"),
    index=False)

selected_features = feature_importance_df.head(TOP_K)["feature"].tolist()
print(f"\nUsing top {TOP_K} features for final SHAP model")

# 13) Fit final model on top-K features
X_train_full_sel = X_train_full[selected_features].copy()
X_test_sel = X_test[selected_features].copy()
imputer_final = SimpleImputer(strategy="median")
X_train_full_imp = imputer_final.fit_transform(X_train_full_sel)
X_test_imp = imputer_final.transform(X_test_sel)
final_model = make_xgb(BEST_PARAMS, random_state=RANDOM_STATE)
final_model.fit(X_train_full_imp, y_train_full, verbose=False)

X_test_imp_df = pd.DataFrame(X_test_imp, columns=selected_features, index=X_test_sel.index)

# 14) SHAP values
print("\nRunning SHAP analysis ...")

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test_imp_df)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_values_df = pd.DataFrame(shap_values, columns=selected_features, index=X_test_imp_df.index)
shap_values_df.index.name = "row_index"
shap_values_df.to_csv(os.path.join(OUTPUT_DIR, "shap_values_test_set.csv"))

# 15) Export SHAP summary CSV for external plotting
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_summary_df = pd.DataFrame({
    "feature": selected_features,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

shap_summary_df["Percentage"] = (
    shap_summary_df["mean_abs_shap"] / shap_summary_df["mean_abs_shap"].sum()
) * 100

def assign_category(feature_name):
    if (
        feature_name in ["BalabanJ", "BertzCT"] or
        feature_name.startswith("Chi") or
        "Kappa" in feature_name or
        feature_name.startswith("Heavy") or
        feature_name.startswith("Num") or
        feature_name.startswith("HallKier")
    ):
        return "Structural"
    elif "VSA" in feature_name:
        return "VSA"
    else:
        return "Other"

shap_summary_df["Category"] = shap_summary_df["feature"].apply(assign_category)

shap_summary_csv = os.path.join(OUTPUT_DIR, "shap_importance_summary.csv")
shap_summary_df.to_csv(shap_summary_csv, index=False)
print(f"Saved SHAP summary CSV to:\n{shap_summary_csv}")

# 16) Export top 20 only
# -------------------------
top20_shap_csv = os.path.join(OUTPUT_DIR, "shap_importance_top20.csv")
shap_summary_df.head(20).to_csv(top20_shap_csv, index=False)
print(f"Saved top-20 SHAP CSV to:\n{top20_shap_csv}")

# 17) SHAP bar plot
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_test_imp_df,
    plot_type="bar",
    max_display=MAX_SHAP_DISPLAY,
    show=False)

fig = plt.gcf()
ax = plt.gca()

ax.set_xlabel("mean(|SHAP value|)", fontsize=18)
ax.set_ylabel("Feature", fontsize=18)
ax.tick_params(axis="both", labelsize=16)

plt.tight_layout()
bar_plot_path = os.path.join(OUTPUT_DIR, "shap_summary_bar.png")
plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved SHAP bar plot to:\n{bar_plot_path}")

# 18) SHAP beeswarm plot
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_test_imp_df,
    max_display=MAX_SHAP_DISPLAY,
    show=False)

fig = plt.gcf()
ax = plt.gca()

ax.set_xlabel("SHAP value (impact on model output)", fontsize=18)
ax.set_ylabel("Feature", fontsize=18)
ax.tick_params(axis="both", labelsize=16)

plt.tight_layout()
beeswarm_plot_path = os.path.join(OUTPUT_DIR, "shap_summary_beeswarm.png")
plt.savefig(beeswarm_plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved SHAP beeswarm plot to:\n{beeswarm_plot_path}")

# 19) Export hold-out metadata + predictions + SHAP indices
test_meta_cols = existing_keep_cols + [SMILES_COL, TARGET_COL]
test_meta = full_df.loc[idx_test, test_meta_cols].reset_index(drop=True)

test_prob = final_model.predict_proba(X_test_imp)[:, 1]
test_pred = (test_prob >= 0.50).astype(int)

pred_df = test_meta.copy()
pred_df.rename(columns={TARGET_COL: "y_true"}, inplace=True)
pred_df["y_prob_toxic"] = test_prob
pred_df["y_pred_label"] = test_pred

pred_df.to_csv(os.path.join(OUTPUT_DIR, "holdout_predictions_for_shap.csv"), index=False)

# 20) Print top SHAP results
print("\nTop 20 SHAP features:")
print(shap_summary_df.head(20))

print(f"\nFiles saved in:\n{OUTPUT_DIR}")
print(f"\nThe End")
