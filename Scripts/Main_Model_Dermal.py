#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:26:53 2026

@author: jfcaetano

# GOOGLE COLAB CODE - XGBoost toxicity model with GO / NO-GO evaluation
# Target: Label (0/1), where 1 = toxic
#
# Decision interpretation:
# - GO = predicted non-toxic (0)
# - NO-GO = predicted toxic (1)
#
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
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    ParameterSampler)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

tqdm.pandas()

# 4) USER SETTINGS
DATA_FILENAME = "/content/drive/MyDrive/ChemEng/acute_dermal.csv"
OUTPUT_DIR = "/content/drive/MyDrive/ChemEng/results_xgboost_go_nogo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMILES_COL = "SMILES"
TARGET_COL = "Label"
KEEP_ID_COLS = ["InChIKey"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_SIZE_WITHIN_TRAIN = 0.20
MAX_ABS_VALUE = 1e10

# Tuning Parameters
N_PARAM_SAMPLES = 30
INNER_CV_FOLDS = 5
REPEATED_CV_SPLITS = 5
REPEATED_CV_REPEATS = 5

# Applicability Doamin
AD_K = 5
AD_PERCENTILE_CUTOFF = 95

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

# 11) Helper functions
def make_xgb(params=None, random_state=42):
    base = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist"
    )
    if params is not None:
        base.update(params)
    return XGBClassifier(**base)

def impute_fit_transform(X_train_part, X_val_part, X_test_part=None):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_part)
    X_val_imp = imputer.transform(X_val_part)
    if X_test_part is not None:
        X_test_imp = imputer.transform(X_test_part)
        return imputer, X_train_imp, X_val_imp, X_test_imp
    return imputer, X_train_imp, X_val_imp

def summarize_results(model_name, y_true, y_pred, y_prob):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_toxic": precision_score(y_true, y_pred, zero_division=0),
        "Recall_toxic": recall_score(y_true, y_pred, zero_division=0),
        "F1_toxic": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob)
    }

def go_no_go_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    predicted_go = (y_pred == 0)
    true_go = (y_true == 0)

    predicted_no_go = (y_pred == 1)
    true_no_go = (y_true == 1)

    go_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    go_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    no_go_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    no_go_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "True_GO": int(tn),          
        "False_NO_GO": int(fp),     
        "False_GO": int(fn),         
        "True_NO_GO": int(tp),      
        "GO_Precision": go_precision,
        "GO_Recall": go_recall,
        "NO_GO_Precision": no_go_precision,
        "NO_GO_Recall": no_go_recall,
        "Toxic_Stopped_Fraction": no_go_recall,
        "NonToxic_Allowed_Fraction": go_recall
    }

def make_go_no_go_confusion_df(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return pd.DataFrame(
        [
            [tn, fp],
            [fn, tp]
        ],
        index=["Actual_GO (non-toxic)", "Actual_NO-GO (toxic)"],
        columns=["Predicted_GO", "Predicted_NO-GO"])

# 12) Parameter search
param_space = {
    "n_estimators": [300, 500, 700, 900],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.07],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 2, 4, 6, 8],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0]
}

sampled_params = list(ParameterSampler(
    param_space,
    n_iter=N_PARAM_SAMPLES,
    random_state=RANDOM_STATE
))

inner_cv = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

search_rows = []
best_score = -np.inf
best_params = None

print("\nManual CV tuning ...")
for i, params in enumerate(sampled_params, start=1):
    fold_scores = []

    for fold_id, (tr_idx, va_idx) in enumerate(inner_cv.split(X_train, y_train), start=1):
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_train.iloc[tr_idx]
        y_va = y_train.iloc[va_idx]

        _, X_tr_imp, X_va_imp = impute_fit_transform(X_tr, X_va)

        model = make_xgb(params=params, random_state=RANDOM_STATE + i + fold_id)
        model.fit(X_tr_imp, y_tr, verbose=False)

        prob_va = model.predict_proba(X_va_imp)[:, 1]
        ap = average_precision_score(y_va, prob_va)
        fold_scores.append(ap)

    mean_ap = float(np.mean(fold_scores))
    std_ap = float(np.std(fold_scores))

    row = {"candidate": i, "mean_AP": mean_ap, "std_AP": std_ap}
    row.update(params)
    search_rows.append(row)

    if mean_ap > best_score:
        best_score = mean_ap
        best_params = params

search_df = pd.DataFrame(search_rows).sort_values("mean_AP", ascending=False).reset_index(drop=True)
search_df.to_csv(os.path.join(OUTPUT_DIR, "manual_cv_search_results.csv"), index=False)

print("\nBest params:")
print(best_params)
print(f"Best mean CV average precision: {best_score:.4f}")

# 13) Train best full model
_, X_train_imp, X_valid_imp = impute_fit_transform(X_train, X_valid)
best_model = make_xgb(best_params, random_state=RANDOM_STATE)
best_model.fit(X_train_imp, y_train, verbose=False)

feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

TOP_K = 60
selected_features = feature_importance_df.head(TOP_K)["feature"].tolist()

print(f"\nUsing top {TOP_K} features for final go/no-go model")

# 14) Fit final model on top-K features
X_train_full_sel = X_train_full[selected_features].copy()
X_test_sel = X_test[selected_features].copy()

imputer_final = SimpleImputer(strategy="median")
X_train_full_imp = imputer_final.fit_transform(X_train_full_sel)
X_test_imp = imputer_final.transform(X_test_sel)

final_model = make_xgb(best_params, random_state=RANDOM_STATE)
final_model.fit(X_train_full_imp, y_train_full, verbose=False)

test_prob = final_model.predict_proba(X_test_imp)[:, 1]

MAIN_THRESHOLD = 0.50
test_pred = (test_prob >= MAIN_THRESHOLD).astype(int)

# 15) Standard classification outputs
results_df = pd.DataFrame([
    summarize_results("XGBoost_top60_0.50", y_test, test_pred, test_prob)
])
results_df.to_csv(os.path.join(OUTPUT_DIR, "standard_results.csv"), index=False)

print("\nStandard classification results:")
print(results_df)

report_df = pd.DataFrame(
    classification_report(y_test, test_pred, output_dict=True, zero_division=0)
).transpose()
report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))

cm_df = pd.DataFrame(
    confusion_matrix(y_test, test_pred),
    index=["Actual_0", "Actual_1"],
    columns=["Pred_0", "Pred_1"]
)
cm_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))

print("\nClassification report:")
print(report_df)

print("\nConfusion matrix:")
print(cm_df)

# 16) GO / NO-GO outputs
gonogo_metrics = go_no_go_metrics(y_test, test_pred)
gonogo_df = pd.DataFrame([gonogo_metrics])
gonogo_df.to_csv(os.path.join(OUTPUT_DIR, "go_no_go_metrics.csv"), index=False)

gonogo_cm_df = make_go_no_go_confusion_df(y_test, test_pred)
gonogo_cm_df.to_csv(os.path.join(OUTPUT_DIR, "go_no_go_confusion_matrix.csv"))

print("\nGO / NO-GO metrics:")
print(gonogo_df)

print("\nGO / NO-GO confusion matrix:")
print(gonogo_cm_df)

# 17) Repeated CV robustness for GO / NO-GO
X_train_full_best = X_train_full[selected_features].copy()

repeated_cv = RepeatedStratifiedKFold(
    n_splits=REPEATED_CV_SPLITS,
    n_repeats=REPEATED_CV_REPEATS,
    random_state=RANDOM_STATE
)

cv_rows = []

print("\nRunning repeated CV for go/no-go robustness ...")
for fold_counter, (tr_idx, va_idx) in enumerate(repeated_cv.split(X_train_full_best, y_train_full), start=1):
    X_cv_tr = X_train_full_best.iloc[tr_idx]
    X_cv_va = X_train_full_best.iloc[va_idx]
    y_cv_tr = y_train_full.iloc[tr_idx]
    y_cv_va = y_train_full.iloc[va_idx]

    _, X_cv_tr_imp, X_cv_va_imp = impute_fit_transform(X_cv_tr, X_cv_va)

    model_cv = make_xgb(best_params, random_state=RANDOM_STATE + fold_counter)
    model_cv.fit(X_cv_tr_imp, y_cv_tr, verbose=False)

    prob_cv = model_cv.predict_proba(X_cv_va_imp)[:, 1]
    pred_cv = (prob_cv >= MAIN_THRESHOLD).astype(int)

    base_metrics = summarize_results("cv_fold", y_cv_va, pred_cv, prob_cv)
    decision_metrics = go_no_go_metrics(y_cv_va, pred_cv)

    row = {}
    row.update(base_metrics)
    row.update(decision_metrics)
    row["Fold"] = fold_counter
    cv_rows.append(row)

cv_results_df = pd.DataFrame(cv_rows)
cv_results_df.to_csv(os.path.join(OUTPUT_DIR, "repeated_cv_go_no_go_results.csv"), index=False)

cv_summary = pd.DataFrame({
    "Metric": [
        "Accuracy", "Precision_toxic", "Recall_toxic", "F1_toxic",
        "ROC_AUC", "PR_AUC", "Brier",
        "GO_Precision", "GO_Recall", "NO_GO_Precision", "NO_GO_Recall"
    ],
    "Mean": [
        cv_results_df["Accuracy"].mean(),
        cv_results_df["Precision_toxic"].mean(),
        cv_results_df["Recall_toxic"].mean(),
        cv_results_df["F1_toxic"].mean(),
        cv_results_df["ROC_AUC"].mean(),
        cv_results_df["PR_AUC"].mean(),
        cv_results_df["Brier"].mean(),
        cv_results_df["GO_Precision"].mean(),
        cv_results_df["GO_Recall"].mean(),
        cv_results_df["NO_GO_Precision"].mean(),
        cv_results_df["NO_GO_Recall"].mean()
    ],
    "Std": [
        cv_results_df["Accuracy"].std(),
        cv_results_df["Precision_toxic"].std(),
        cv_results_df["Recall_toxic"].std(),
        cv_results_df["F1_toxic"].std(),
        cv_results_df["ROC_AUC"].std(),
        cv_results_df["PR_AUC"].std(),
        cv_results_df["Brier"].std(),
        cv_results_df["GO_Precision"].std(),
        cv_results_df["GO_Recall"].std(),
        cv_results_df["NO_GO_Precision"].std(),
        cv_results_df["NO_GO_Recall"].std()
    ]
})
cv_summary.to_csv(os.path.join(OUTPUT_DIR, "repeated_cv_go_no_go_summary.csv"), index=False)

print("\nRepeated CV GO / NO-GO summary:")
print(cv_summary)

# 18) Applicability domain
imputer_ad = SimpleImputer(strategy="median")
scaler_ad = StandardScaler()

X_train_imp_ad = imputer_ad.fit_transform(X_train_full_best)
X_test_imp_ad = imputer_ad.transform(X_test_sel)

X_train_scaled = scaler_ad.fit_transform(X_train_imp_ad)
X_test_scaled = scaler_ad.transform(X_test_imp_ad)

nn_train = NearestNeighbors(n_neighbors=AD_K + 1, metric="euclidean")
nn_train.fit(X_train_scaled)
train_dist, _ = nn_train.kneighbors(X_train_scaled)
train_mean_k = train_dist[:, 1:(AD_K + 1)].mean(axis=1)

nn_test = NearestNeighbors(n_neighbors=AD_K, metric="euclidean")
nn_test.fit(X_train_scaled)
test_dist, _ = nn_test.kneighbors(X_test_scaled)
test_mean_k = test_dist.mean(axis=1)

ad_cutoff = np.percentile(train_mean_k, AD_PERCENTILE_CUTOFF)
test_in_ad = test_mean_k <= ad_cutoff

print(f"\nApplicability domain cutoff: {ad_cutoff:.4f}")
print(f"Test compounds inside AD: {int(test_in_ad.sum())} / {len(test_in_ad)}")

ad_rows = []
for subset_name, mask in [("Inside_AD", test_in_ad), ("Outside_AD", ~test_in_ad)]:
    if mask.sum() > 0:
        y_sub = y_test[mask]
        prob_sub = test_prob[mask]
        pred_sub = test_pred[mask]

        row = summarize_results(subset_name, y_sub, pred_sub, prob_sub)
        row.update(go_no_go_metrics(y_sub, pred_sub))
        row["Subset"] = subset_name
        row["N"] = int(mask.sum())
        ad_rows.append(row)

ad_df = pd.DataFrame(ad_rows)
ad_df.to_csv(os.path.join(OUTPUT_DIR, "applicability_domain_go_no_go_results.csv"), index=False)

print("\nApplicability domain GO / NO-GO results:")
print(ad_df)

# 19) Save holdout predictions
test_meta_cols = existing_keep_cols + [SMILES_COL, TARGET_COL]
test_meta = full_df.loc[idx_test, test_meta_cols].reset_index(drop=True)

pred_df = test_meta.copy()
pred_df.rename(columns={TARGET_COL: "y_true"}, inplace=True)
pred_df["Decision"] = np.where(test_pred == 1, "NO-GO", "GO")
pred_df["y_prob_toxic"] = test_prob
pred_df["y_pred_label"] = test_pred
pred_df["mean_kNN_distance"] = test_mean_k
pred_df["inside_applicability_domain"] = test_in_ad

pred_df.to_csv(os.path.join(OUTPUT_DIR, "holdout_predictions_go_no_go.csv"), index=False)

# 20) Results summary
tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

summary_text = f"""
GO / NO-GO DECISION SUMMARY

Interpretation:
- GO = predicted non-toxic
- NO-GO = predicted toxic

Test-set results:
- True GO (non-toxic correctly allowed): {tn}
- False NO-GO (non-toxic incorrectly blocked): {fp}
- False GO (toxic incorrectly allowed): {fn}
- True NO-GO (toxic correctly blocked): {tp}

Key screening interpretation:
- Fraction of toxic compounds stopped (NO-GO recall): {tp / (tp + fn):.3f}
- Fraction of non-toxic compounds allowed (GO recall): {tn / (tn + fp):.3f}
- Reliability of NO-GO decisions (NO-GO precision): {tp / (tp + fp) if (tp + fp) > 0 else 0:.3f}
- Reliability of GO decisions (GO precision): {tn / (tn + fn) if (tn + fn) > 0 else 0:.3f}
"""

with open(os.path.join(OUTPUT_DIR, "go_no_go_summary.txt"), "w") as f:
    f.write(summary_text)

print("\n" + summary_text)
print(f"\nFiles saved in:\n{OUTPUT_DIR}")
print(f"\nThe End")
