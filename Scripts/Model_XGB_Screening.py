#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:01:09 2026

@author: jfcaetano

# GOOGLE COLAB: Stronger XGBoost-only toxicity screening workflow
# Target: Label (0/1), where 1 = toxic
#
# Improvements vs previous version:
# - more CV during tuning
# - repeated CV robustness
# - broader hyperparameter search
# - top-K feature subset comparison
# - early stopping inside fit
# - corrected applicability domain
#
# Main screening model threshold determination

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
from sklearn.pipeline import Pipeline
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
OUTPUT_DIR = "/content/drive/MyDrive/ChemEng/results_xgboost_sustainability_stronger"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMILES_COL = "SMILES"
TARGET_COL = "Label"
KEEP_ID_COLS = ["InChIKey"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_SIZE_WITHIN_TRAIN = 0.20
MAX_ABS_VALUE = 1e10

N_PARAM_SAMPLES = 30
INNER_CV_FOLDS = 5
REPEATED_CV_SPLITS = 5
REPEATED_CV_REPEATS = 5

THRESHOLDS = np.arange(0.30, 0.71, 0.02)
MIN_TOXIC_PRECISION = 0.58

# Feature subset comparison
TOP_K_OPTIONS = [40, 60, 80, 110]
EARLY_STOPPING_ROUNDS = 40

# Applicability domain
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


# 10) Train / validation / test split

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
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob)
    }

def choose_threshold(y_true, y_prob, thresholds, min_precision=0.58):
    rows = []
    best_thr = 0.50
    best_score = -1

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

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


# 12) Parameter search with manual CV + early stopping

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

print("\nManual CV tuning with early stopping ...")
for i, params in enumerate(sampled_params, start=1):
    fold_scores = []

    for fold_id, (tr_idx, va_idx) in enumerate(inner_cv.split(X_train, y_train), start=1):
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_train.iloc[tr_idx]
        y_va = y_train.iloc[va_idx]

        _, X_tr_imp, X_va_imp = impute_fit_transform(X_tr, X_va)

        model = make_xgb(params=params, random_state=RANDOM_STATE + i + fold_id)
        model.fit(
            X_tr_imp, y_tr,
            eval_set=[(X_va_imp, y_va)],
            verbose=False
        )

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


# 13) Feature importance ranking from best full model

_, X_train_imp, X_valid_imp = impute_fit_transform(X_train, X_valid)
best_full_model = make_xgb(best_params, random_state=RANDOM_STATE)
best_full_model.fit(
    X_train_imp, y_train,
    eval_set=[(X_valid_imp, y_valid)],
    verbose=False)

feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": best_full_model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_full.csv"), index=False)

print("\nTop 20 features:")
print(feature_importance_df.head(20))


# 14) Compare feature subsets

candidate_rows = []
candidate_objects = []

feature_sets = []
for k in TOP_K_OPTIONS:
    kk = min(k, X.shape[1])
    features = feature_importance_df.head(kk)["feature"].tolist()
    feature_sets.append((f"top{kk}", features))

for set_name, feature_list in feature_sets:
    print(f"\nEvaluating feature set: {set_name} ({len(feature_list)} descriptors)")

    X_tr_fs = X_train[feature_list]
    X_va_fs = X_valid[feature_list]
    X_tr_full_fs = X_train_full[feature_list]
    X_te_fs = X_test[feature_list]

    _, X_tr_imp, X_va_imp, X_te_imp = impute_fit_transform(X_tr_fs, X_va_fs, X_te_fs)

    model_fs = make_xgb(best_params, random_state=RANDOM_STATE)
    model_fs.fit(
        X_tr_imp, y_train,
        eval_set=[(X_va_imp, y_valid)],
        verbose=False
    )

    valid_prob = model_fs.predict_proba(X_va_imp)[:, 1]
    best_thr, thr_score, thr_df = choose_threshold(
        y_valid, valid_prob, THRESHOLDS, MIN_TOXIC_PRECISION
    )
    thr_df.to_csv(os.path.join(OUTPUT_DIR, f"threshold_search_{set_name}.csv"), index=False)

    # refit on train+valid using same features
    imputer_final = SimpleImputer(strategy="median")
    X_tr_full_imp = imputer_final.fit_transform(X_tr_full_fs)
    X_te_imp_final = imputer_final.transform(X_te_fs)

    model_final = make_xgb(best_params, random_state=RANDOM_STATE)
    model_final.fit(X_tr_full_imp, y_train_full, verbose=False)

    test_prob = model_final.predict_proba(X_te_imp_final)[:, 1]
    test_pred_050 = (test_prob >= 0.50).astype(int)
    test_pred_tuned = (test_prob >= best_thr).astype(int)

    row_default = summarize_results(f"XGBoost_{set_name}_0.50", y_test, test_pred_050, test_prob)
    row_default["Threshold"] = 0.50
    row_default["Feature_Count"] = len(feature_list)

    row_tuned = summarize_results(f"XGBoost_{set_name}_{best_thr:.2f}", y_test, test_pred_tuned, test_prob)
    row_tuned["Threshold"] = best_thr
    row_tuned["Feature_Count"] = len(feature_list)

    candidate_rows.extend([row_default, row_tuned])

    candidate_objects.append({
        "set_name": set_name,
        "feature_list": feature_list,
        "best_thr": best_thr,
        "imputer": imputer_final,
        "model": model_final,
        "test_prob": test_prob,
        "test_pred_050": test_pred_050,
        "test_pred_tuned": test_pred_tuned
    })

results_df = pd.DataFrame(candidate_rows)
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

results_df.to_csv(os.path.join(OUTPUT_DIR, "candidate_results_ranked.csv"), index=False)

print("\nRanked candidate results:")
print(results_df)

best_name = results_df.iloc[0]["Model"]
print(f"\nBest candidate: {best_name}")

best_obj = None
for obj in candidate_objects:
    n1 = f"XGBoost_{obj['set_name']}_0.50"
    n2 = f"XGBoost_{obj['set_name']}_{obj['best_thr']:.2f}"
    if best_name in [n1, n2]:
        best_obj = obj
        break

if best_obj is None:
    raise RuntimeError("Best model object not found.")

if best_name.endswith("_0.50"):
    final_pred = best_obj["test_pred_050"]
else:
    final_pred = best_obj["test_pred_tuned"]

final_prob = best_obj["test_prob"]
final_features = best_obj["feature_list"]


# 15) Repeated CV robustness on chosen feature subset
X_train_full_best = X_train_full[final_features].copy()

repeated_cv = RepeatedStratifiedKFold(
    n_splits=REPEATED_CV_SPLITS,
    n_repeats=REPEATED_CV_REPEATS,
    random_state=RANDOM_STATE
)

cv_rows = []
fold_counter = 0

print("\nRunning repeated CV on chosen feature subset ...")
for tr_idx, va_idx in repeated_cv.split(X_train_full_best, y_train_full):
    fold_counter += 1

    X_cv_tr = X_train_full_best.iloc[tr_idx]
    X_cv_va = X_train_full_best.iloc[va_idx]
    y_cv_tr = y_train_full.iloc[tr_idx]
    y_cv_va = y_train_full.iloc[va_idx]

    _, X_cv_tr_imp, X_cv_va_imp = impute_fit_transform(X_cv_tr, X_cv_va)

    model_cv = make_xgb(best_params, random_state=RANDOM_STATE + fold_counter)
    model_cv.fit(X_cv_tr_imp, y_cv_tr, verbose=False)

    prob_cv = model_cv.predict_proba(X_cv_va_imp)[:, 1]
    pred_cv = (prob_cv >= 0.50).astype(int)

    cv_rows.append({
        "Fold": fold_counter,
        "Accuracy": accuracy_score(y_cv_va, pred_cv),
        "Precision": precision_score(y_cv_va, pred_cv, zero_division=0),
        "Recall": recall_score(y_cv_va, pred_cv, zero_division=0),
        "F1": f1_score(y_cv_va, pred_cv, zero_division=0),
        "ROC_AUC": roc_auc_score(y_cv_va, prob_cv),
        "PR_AUC": average_precision_score(y_cv_va, prob_cv),
        "Brier": brier_score_loss(y_cv_va, prob_cv)
    })

cv_results_df = pd.DataFrame(cv_rows)
cv_results_df.to_csv(os.path.join(OUTPUT_DIR, "repeated_cv_results_best_subset.csv"), index=False)

cv_summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "Brier"],
    "Mean": [
        cv_results_df["Accuracy"].mean(),
        cv_results_df["Precision"].mean(),
        cv_results_df["Recall"].mean(),
        cv_results_df["F1"].mean(),
        cv_results_df["ROC_AUC"].mean(),
        cv_results_df["PR_AUC"].mean(),
        cv_results_df["Brier"].mean()
    ],
    "Std": [
        cv_results_df["Accuracy"].std(),
        cv_results_df["Precision"].std(),
        cv_results_df["Recall"].std(),
        cv_results_df["F1"].std(),
        cv_results_df["ROC_AUC"].std(),
        cv_results_df["PR_AUC"].std(),
        cv_results_df["Brier"].std()
    ]
})
cv_summary.to_csv(os.path.join(OUTPUT_DIR, "repeated_cv_summary_best_subset.csv"), index=False)

print("\nRepeated CV summary on best subset:")
print(cv_summary)


# 16) Applicability domain on chosen feature subset
X_test_best = X_test[final_features].copy()

imputer_ad = SimpleImputer(strategy="median")
scaler_ad = StandardScaler()

X_train_imp_ad = imputer_ad.fit_transform(X_train_full_best)
X_test_imp_ad = imputer_ad.transform(X_test_best)

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
    if mask.sum() > 0 and len(np.unique(y_test[mask])) == 2:
        y_sub = y_test[mask]
        p_sub = final_prob[mask]
        pred_sub = final_pred[mask]

        ad_rows.append({
            "Subset": subset_name,
            "N": int(mask.sum()),
            "Accuracy": accuracy_score(y_sub, pred_sub),
            "Precision": precision_score(y_sub, pred_sub, zero_division=0),
            "Recall": recall_score(y_sub, pred_sub, zero_division=0),
            "F1": f1_score(y_sub, pred_sub, zero_division=0),
            "ROC_AUC": roc_auc_score(y_sub, p_sub),
            "PR_AUC": average_precision_score(y_sub, p_sub)
        })

ad_results_df = pd.DataFrame(ad_rows)
ad_results_df.to_csv(os.path.join(OUTPUT_DIR, "applicability_domain_performance.csv"), index=False)

print("\nApplicability domain performance:")
print(ad_results_df)


# 17) Final outputs
test_meta_cols = existing_keep_cols + [SMILES_COL, TARGET_COL]
test_meta = full_df.loc[idx_test, test_meta_cols].reset_index(drop=True)

pred_df = test_meta.copy()
pred_df.rename(columns={TARGET_COL: "y_true"}, inplace=True)
pred_df["y_prob"] = final_prob
pred_df["y_pred"] = final_pred
pred_df["mean_kNN_distance"] = test_mean_k
pred_df["inside_applicability_domain"] = test_in_ad
pred_df.to_csv(os.path.join(OUTPUT_DIR, "holdout_predictions_best_model.csv"), index=False)

final_report = pd.DataFrame(
    classification_report(y_test, final_pred, output_dict=True, zero_division=0)
).transpose()
final_report.to_csv(os.path.join(OUTPUT_DIR, "best_model_classification_report.csv"))

final_cm = pd.DataFrame(
    confusion_matrix(y_test, final_pred),
    index=["Actual_0", "Actual_1"],
    columns=["Pred_0", "Pred_1"]
)
final_cm.to_csv(os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.csv"))

print("\nBest final classification report:")
print(final_report)

print("\nBest final confusion matrix:")
print(final_cm)

print("\nFiles saved in:")
print(OUTPUT_DIR)
print(f"\nThe End")
