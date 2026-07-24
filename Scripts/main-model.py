#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 23:22:42 2026

@author: jfcaetano
"""

# ============================================================
# GOOGLE COLAB: END-TO-END SCAFFOLD-LOCKED TOXICITY WORKFLOW
#
# Target: Label (0/1), where 1 = toxic
#
# Decision interpretation:
# - GO = predicted non-toxic (0)
# - NO-GO = predicted toxic (1)
#
# WORKFLOW
# 1. Clean and standardize structures.
# 2. Create a new locked scaffold test partition.
# 3. Compare Logistic Regression, SVM, RF, GB, and XGBoost on
#    development data only using identical scaffold-grouped folds.
# 4. Tune weighted XGBoost on development data only.
# 5. Select/freeze the top 60 descriptors.
# 6. Calculate VIF for the frozen descriptors.
# 7. Freeze the operating threshold from weighted development OOF
#    predictions.
# 8. Compare scale_pos_weight, SMOTE, and ADASYN using the same
#    frozen descriptors, hyperparameters, scaffold folds, and threshold.
# 9. Evaluate all three imbalance approaches on the locked test once.
# 10. Run applicability-domain and SHAP analyses on the original
#     weighted final model, with VIF annotations.
#
# RUN MODES
# - RUN_MODE = "development"
#     Creates the split, compares model families, tunes XGBoost,
#     freezes descriptors and threshold, calculates VIF, and performs
#     a development-only OOF comparison of scale_pos_weight, SMOTE,
#     and ADASYN.
#     It DOES NOT evaluate the locked test.
#
# - RUN_MODE = "final_test"
#     Loads frozen development artifacts, fits the exact weighted,
#     SMOTE, and ADASYN models on all development compounds, evaluates the locked
#     test once, and runs AD + SHAP on the weighted final model.
#
# LOGGING
# - Console output and tqdm progress bars remain visible in Colab.
# - The same console output is copied to:
#       workflow_<RUN_MODE>.log
# - A structured stage/event log is copied to:
#       workflow_events_<RUN_MODE>.csv
# - The original data-cleaning removal log remains unchanged.
#
# IMPORTANT
# Run "development" first. Review and archive the frozen files.
# Then change only RUN_MODE to "final_test" and rerun the notebook
# from the beginning. Do not retune after final-test evaluation.
# ============================================================

# -------------------------
# 1) Install dependencies
# -------------------------
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

# -------------------------
# 2) Mount Google Drive
# -------------------------
from google.colab import drive

drive.mount("/content/drive")

# -------------------------
# 3) Imports
# -------------------------
import csv
import json
import os
import platform
import traceback
import warnings
from datetime import datetime, timezone

import imblearn
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost

from tqdm.auto import tqdm

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    ParameterSampler,
    StratifiedGroupKFold,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbalancedPipeline

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
tqdm.pandas()

# -------------------------
# 4) USER SETTINGS
# -------------------------
DATA_FILENAME = "/content/drive/MyDrive/ChemEng/acute_dermal.csv"

# A dedicated directory prevents outputs from earlier workflows
# from being silently reused.
OUTPUT_DIR = (
    "/content/drive/MyDrive/ChemEng/"
    "results_scaffold_locked_qsar_vif_smote_adasyn"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run development first, then change only this value to "final_test".
RUN_MODE = "final_test"  # "development" or "final_test"

# Prevent accidental repeated opening of the locked test.
ALLOW_REPEAT_FINAL_TEST = True

# Run the resampling comparisons in both modes.
RUN_SMOTE_COMPARISON = True
RUN_ADASYN_COMPARISON = True

SMILES_COL = "SMILES"
TARGET_COL = "Label"
KEEP_ID_COLS = ["InChIKey"]

RANDOM_STATE = 42
MAX_ABS_VALUE = 1e10

# Locked scaffold test.
TEST_SIZE = 0.20
N_SPLIT_CANDIDATES = 500

# Algorithm-family comparison.
ALGORITHM_CV_FOLDS = 5
BASELINE_THRESHOLD = 0.50

# XGBoost hyperparameter tuning.
N_PARAM_SAMPLES = 30
TUNING_CV_FOLDS = 5

# Feature selection, VIF, and threshold.
TOP_K = 60
VIF_WARNING_THRESHOLD = 5.0
VIF_SEVERE_THRESHOLD = 10.0
THRESHOLD_GRID = np.arange(0.20, 0.81, 0.02)
MIN_TOXIC_PRECISION = 0.58
DEFAULT_THRESHOLD = 0.50

# SMOTE settings.
SMOTE_K_NEIGHBORS = 5
SMOTE_SAMPLING_STRATEGY = "auto"

# ADASYN settings.
ADASYN_N_NEIGHBORS = 5
ADASYN_SAMPLING_STRATEGY = "auto"

# Applicability domain.
AD_K = 5
AD_PERCENTILE_CUTOFF = 95

# Reproducibility.
N_JOBS = 1
np.random.seed(RANDOM_STATE)

# Artifact paths.
CLEANED_DATA_PATH = os.path.join(
    OUTPUT_DIR,
    "cleaned_standardized_dataset.csv",
)
REMOVAL_LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "data_cleaning_removal_log.csv",
)
SPLIT_PATH = os.path.join(
    OUTPUT_DIR,
    "locked_scaffold_split_assignments.csv",
)
ALGORITHM_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "chunk1_algorithm_comparison.csv",
)
ALGORITHM_FOLD_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "chunk1_algorithm_comparison_folds.csv",
)
ALGORITHM_OOF_PATH = os.path.join(
    OUTPUT_DIR,
    "chunk1_algorithm_comparison_oof_predictions.csv",
)
TUNING_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "chunk2_xgboost_search_results.csv",
)
FROZEN_CONFIG_PATH = os.path.join(
    OUTPUT_DIR,
    "frozen_model_configuration.json",
)
SELECTED_FEATURES_PATH = os.path.join(
    OUTPUT_DIR,
    "frozen_top60_features.csv",
)
VIF_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "development_top60_vif.csv",
)
IMPORTANCE_WITH_VIF_PATH = os.path.join(
    OUTPUT_DIR,
    "development_top60_importance_with_vif.csv",
)
DEVELOPMENT_OOF_PATH = os.path.join(
    OUTPUT_DIR,
    "development_oof_predictions.csv",
)
FINAL_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "final_frozen_pipeline.joblib",
)
FINAL_FLAG_PATH = os.path.join(
    OUTPUT_DIR,
    "FINAL_TEST_EVALUATED.flag",
)
IMBALANCE_OOF_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "development_imbalance_method_comparison.csv",
)
IMBALANCE_OOF_PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "development_imbalance_oof_predictions.csv",
)
IMBALANCE_TEST_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "locked_test_imbalance_method_comparison.csv",
)
SMOTE_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "final_smote_pipeline.joblib",
)
ADASYN_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "final_adasyn_pipeline.joblib",
)

# Run logs.
RUN_LOG_PATH = os.path.join(
    OUTPUT_DIR,
    f"workflow_{RUN_MODE}.log",
)
RUN_EVENT_LOG_PATH = os.path.join(
    OUTPUT_DIR,
    f"workflow_events_{RUN_MODE}.csv",
)


class TeeStream:
    """Copy console output to both Colab and a UTF-8 log file."""

    def __init__(self, original_stream, log_stream):
        self.original_stream = original_stream
        self.log_stream = log_stream

    def write(self, message):
        self.original_stream.write(message)
        self.log_stream.write(message)
        return len(message)

    def flush(self):
        self.original_stream.flush()
        self.log_stream.flush()

    def isatty(self):
        try:
            return self.original_stream.isatty()
        except Exception:
            return False

    def fileno(self):
        return self.original_stream.fileno()

    @property
    def encoding(self):
        return getattr(self.original_stream, "encoding", "utf-8")


# Avoid nested tee streams when the notebook is rerun in the same kernel.
if isinstance(sys.stdout, TeeStream):
    sys.stdout = sys.stdout.original_stream
if isinstance(sys.stderr, TeeStream):
    sys.stderr = sys.stderr.original_stream

_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_RUN_LOG_HANDLE = open(
    RUN_LOG_PATH,
    "a",
    encoding="utf-8",
    buffering=1,
)
sys.stdout = TeeStream(
    _ORIGINAL_STDOUT,
    _RUN_LOG_HANDLE,
)
sys.stderr = TeeStream(
    _ORIGINAL_STDERR,
    _RUN_LOG_HANDLE,
)


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def log_event(stage, status, message=""):
    """Append one structured workflow event to a CSV file."""
    file_exists = os.path.exists(RUN_EVENT_LOG_PATH)
    row = {
        "Timestamp_UTC": utc_timestamp(),
        "Run_Mode": RUN_MODE,
        "Stage": stage,
        "Status": status,
        "Message": str(message),
    }

    with open(
        RUN_EVENT_LOG_PATH,
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row.keys()),
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


print("\n" + "=" * 72)
print("SCAFFOLD-LOCKED TOXICITY WORKFLOW")
print("Run mode:", RUN_MODE)
print("Started (UTC):", utc_timestamp())
print("Console log:", RUN_LOG_PATH)
print("Structured event log:", RUN_EVENT_LOG_PATH)
print("=" * 72)
log_event("workflow", "started", f"RUN_MODE={RUN_MODE}")

# -------------------------
# 5) RDKit descriptor selection
# -------------------------
descriptor_names = []

for descriptor_name in dir(Descriptors):
    if descriptor_name in ["BalabanJ", "BertzCT", "TPSA"]:
        descriptor_names.append(descriptor_name)
    elif descriptor_name[:3] == "Chi":
        descriptor_names.append(descriptor_name)
    elif "VSA" in descriptor_name:
        descriptor_names.append(descriptor_name)
    elif "Kappa" in descriptor_name:
        descriptor_names.append(descriptor_name)
    elif descriptor_name[:1] == "H":
        descriptor_names.append(descriptor_name)
    elif descriptor_name[:1] == "N":
        descriptor_names.append(descriptor_name)
    elif descriptor_name[:1] == "M":
        descriptor_names.append(descriptor_name)

SELECTED_DESCRIPTORS = sorted({
    descriptor_name
    for descriptor_name in descriptor_names
    if hasattr(Descriptors, descriptor_name)
    and callable(getattr(Descriptors, descriptor_name))
})

print(
    "Number of initially selected RDKit descriptors:",
    len(SELECTED_DESCRIPTORS),
)

# -------------------------
# 6) Structure and descriptor helpers
# -------------------------
def canonicalize_smiles(smiles):
    if pd.isna(smiles) or str(smiles).strip() == "":
        return None

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None

    try:
        return Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None


def scaffold_from_canonical_smiles(canonical_smiles):
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)

    if scaffold is None or scaffold.GetNumAtoms() == 0:
        # Acyclic compounds otherwise collapse to one empty scaffold.
        return f"ACYCLIC::{canonical_smiles}"

    return Chem.MolToSmiles(
        scaffold,
        canonical=True,
        isomericSmiles=False,
    )


def smiles_to_descriptor_dict(canonical_smiles):
    if canonical_smiles is None:
        return {
            descriptor_name: np.nan
            for descriptor_name in SELECTED_DESCRIPTORS
        }

    molecule = Chem.MolFromSmiles(canonical_smiles)

    if molecule is None:
        return {
            descriptor_name: np.nan
            for descriptor_name in SELECTED_DESCRIPTORS
        }

    descriptor_values = {}

    for descriptor_name in SELECTED_DESCRIPTORS:
        try:
            descriptor_function = getattr(
                Descriptors,
                descriptor_name,
            )
            value = descriptor_function(molecule)

            if value is None:
                value = np.nan
            elif isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                if not np.isfinite(value):
                    value = np.nan
                elif abs(value) > MAX_ABS_VALUE:
                    value = np.nan

            descriptor_values[descriptor_name] = value

        except Exception:
            descriptor_values[descriptor_name] = np.nan

    return descriptor_values


def calculate_positive_weight(labels):
    negative_count = int((labels == 0).sum())
    positive_count = int((labels == 1).sum())

    if positive_count == 0:
        return 1.0

    return negative_count / positive_count


def make_group_cv(n_splits, random_state):
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def safe_binary_metrics(y_true, y_pred, y_prob):
    """Return threshold-dependent and probability-based binary metrics."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_prob_array = np.asarray(y_prob, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "Accuracy": accuracy_score(
            y_true_array,
            y_pred_array,
        ),
        "BalancedAccuracy": balanced_accuracy_score(
            y_true_array,
            y_pred_array,
        ),
        "Precision_toxic": precision_score(
            y_true_array,
            y_pred_array,
            zero_division=0,
        ),
        "Recall_toxic": recall_score(
            y_true_array,
            y_pred_array,
            zero_division=0,
        ),
        "Specificity_non_toxic": specificity,
        "F1_toxic": f1_score(
            y_true_array,
            y_pred_array,
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(
            y_true_array,
            y_pred_array,
        ),
        "Brier": brier_score_loss(
            y_true_array,
            y_prob_array,
        ),
    }

    if len(np.unique(y_true_array)) == 2:
        metrics["ROC_AUC"] = roc_auc_score(
            y_true_array,
            y_prob_array,
        )
        metrics["PR_AUC"] = average_precision_score(
            y_true_array,
            y_prob_array,
        )
    else:
        metrics["ROC_AUC"] = np.nan
        metrics["PR_AUC"] = np.nan

    return metrics

def go_no_go_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "True_GO": int(tn),
        "False_NO_GO": int(fp),
        "False_GO": int(fn),
        "True_NO_GO": int(tp),
        "GO_Precision": (
            tn / (tn + fn)
            if (tn + fn) > 0
            else 0.0
        ),
        "GO_Recall": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        ),
        "NO_GO_Precision": (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        ),
        "NO_GO_Recall": (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        ),
    }


def choose_threshold(
    y_true,
    y_probability,
    threshold_grid,
    minimum_precision,
):
    rows = []
    eligible_rows = []

    for threshold in threshold_grid:
        prediction = (
            y_probability >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            prediction,
            zero_division=0,
        )
        recall = recall_score(
            y_true,
            prediction,
            zero_division=0,
        )
        f1 = f1_score(
            y_true,
            prediction,
            zero_division=0,
        )

        row = {
            "Threshold": float(threshold),
            "Precision_toxic": precision,
            "Recall_toxic": recall,
            "F1_toxic": f1,
        }
        rows.append(row)

        if precision >= minimum_precision:
            eligible_rows.append(row)

    threshold_table = pd.DataFrame(rows)

    if eligible_rows:
        best_row = sorted(
            eligible_rows,
            key=lambda row: (
                row["F1_toxic"],
                row["Recall_toxic"],
                -abs(row["Threshold"] - DEFAULT_THRESHOLD),
            ),
            reverse=True,
        )[0]
        selected_threshold = best_row["Threshold"]
    else:
        selected_threshold = DEFAULT_THRESHOLD

    return float(selected_threshold), threshold_table


def make_algorithm_pipeline(
    algorithm_name,
    positive_weight,
    random_state,
):
    requires_scaling = False

    if algorithm_name == "LogisticRegression":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
        requires_scaling = True

    elif algorithm_name == "SVM":
        model = SVC(
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )
        requires_scaling = True

    elif algorithm_name == "RandomForest":
        model = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=N_JOBS,
        )

    elif algorithm_name == "GradientBoosting":
        model = GradientBoostingClassifier(
            random_state=random_state,
        )

    elif algorithm_name == "XGBoost":
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=positive_weight,
            tree_method="hist",
            random_state=random_state,
            n_jobs=N_JOBS,
        )

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}"
        )

    pipeline_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]

    if requires_scaling:
        pipeline_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    pipeline_steps.append(
        (
            "model",
            model,
        )
    )

    return Pipeline(pipeline_steps)

def make_xgb_pipeline(
    params,
    positive_weight,
    random_state,
):
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=positive_weight,
        tree_method="hist",
        random_state=random_state,
        n_jobs=N_JOBS,
        **params,
    )

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            model,
        ),
    ])


def make_smote_xgb_pipeline(
    params,
    minority_sample_count,
    random_state,
):
    """
    Construct an XGBoost pipeline using SMOTE instead of
    scale_pos_weight.

    SMOTE is applied only during pipeline fitting. Validation and
    locked-test observations are never resampled.
    """

    if minority_sample_count < 2:
        raise ValueError(
            "SMOTE requires at least two toxic compounds "
            "in the training partition."
        )

    effective_k_neighbors = min(
        SMOTE_K_NEIGHBORS,
        minority_sample_count - 1,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=1.0,
        tree_method="hist",
        random_state=random_state,
        n_jobs=N_JOBS,
        **params,
    )

    return ImbalancedPipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "smote",
            SMOTE(
                sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                k_neighbors=effective_k_neighbors,
                random_state=random_state,
            ),
        ),
        (
            "model",
            model,
        ),
    ])


def make_adasyn_xgb_pipeline(
    params,
    minority_sample_count,
    random_state,
):
    """
    Construct an XGBoost pipeline using ADASYN instead of
    scale_pos_weight.

    ADASYN is applied only during pipeline fitting. Validation and
    locked-test observations are never resampled. Median imputation
    and scaling are fitted exclusively on each training partition.
    """

    if minority_sample_count < 2:
        raise ValueError(
            "ADASYN requires at least two toxic compounds "
            "in the training partition."
        )

    effective_n_neighbors = min(
        ADASYN_N_NEIGHBORS,
        minority_sample_count - 1,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=1.0,
        tree_method="hist",
        random_state=random_state,
        n_jobs=N_JOBS,
        **params,
    )

    return ImbalancedPipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "adasyn",
            ADASYN(
                sampling_strategy=ADASYN_SAMPLING_STRATEGY,
                n_neighbors=effective_n_neighbors,
                random_state=random_state,
            ),
        ),
        (
            "model",
            model,
        ),
    ])

# -------------------------
# 7) Clean and standardize dataset
# -------------------------
def build_cleaned_dataset():
    raw_df = pd.read_csv(DATA_FILENAME)
    print("Loaded raw dataset shape:", raw_df.shape)

    required_columns = [SMILES_COL, TARGET_COL]
    missing_columns = [
        column
        for column in required_columns
        if column not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    existing_id_columns = [
        column
        for column in KEEP_ID_COLS
        if column in raw_df.columns
    ]

    working_df = raw_df[
        existing_id_columns
        + [SMILES_COL, TARGET_COL]
    ].copy()

    working_df["Original_Row"] = raw_df.index
    working_df[TARGET_COL] = pd.to_numeric(
        working_df[TARGET_COL],
        errors="coerce",
    )

    removal_rows = []

    invalid_label_mask = (
        working_df[TARGET_COL].isna()
        | ~working_df[TARGET_COL].isin([0, 1])
    )

    for _, row in working_df[
        invalid_label_mask
    ].iterrows():
        removal_rows.append({
            "Original_Row": row["Original_Row"],
            "Reason": "Invalid or missing label",
            "SMILES": row[SMILES_COL],
        })

    working_df = working_df[
        ~invalid_label_mask
    ].copy()
    working_df[TARGET_COL] = working_df[
        TARGET_COL
    ].astype(int)

    print("\nCanonicalizing structures ...")
    working_df["Canonical_SMILES"] = working_df[
        SMILES_COL
    ].progress_apply(canonicalize_smiles)

    invalid_smiles_mask = working_df[
        "Canonical_SMILES"
    ].isna()

    for _, row in working_df[
        invalid_smiles_mask
    ].iterrows():
        removal_rows.append({
            "Original_Row": row["Original_Row"],
            "Reason": "Invalid or empty structure",
            "SMILES": row[SMILES_COL],
        })

    working_df = working_df[
        ~invalid_smiles_mask
    ].copy()

    label_counts_by_structure = (
        working_df.groupby(
            "Canonical_SMILES"
        )[TARGET_COL]
        .nunique()
    )

    conflicting_structures = set(
        label_counts_by_structure[
            label_counts_by_structure > 1
        ].index
    )

    conflict_mask = working_df[
        "Canonical_SMILES"
    ].isin(conflicting_structures)

    for _, row in working_df[
        conflict_mask
    ].iterrows():
        removal_rows.append({
            "Original_Row": row["Original_Row"],
            "Reason": "Canonical structure has conflicting labels",
            "SMILES": row[SMILES_COL],
        })

    working_df = working_df[
        ~conflict_mask
    ].copy()

    duplicate_mask = working_df.duplicated(
        subset=["Canonical_SMILES"],
        keep="first",
    )

    for _, row in working_df[
        duplicate_mask
    ].iterrows():
        removal_rows.append({
            "Original_Row": row["Original_Row"],
            "Reason": "Duplicate canonical structure with same label",
            "SMILES": row[SMILES_COL],
        })

    working_df = working_df[
        ~duplicate_mask
    ].copy().reset_index(drop=True)

    working_df["Scaffold"] = working_df[
        "Canonical_SMILES"
    ].apply(scaffold_from_canonical_smiles)

    print("\nCalculating RDKit descriptors ...")
    descriptor_df = working_df[
        "Canonical_SMILES"
    ].progress_apply(
        lambda value: pd.Series(
            smiles_to_descriptor_dict(value)
        )
    )

    cleaned_df = pd.concat(
        [
            working_df.reset_index(drop=True),
            descriptor_df.reset_index(drop=True),
        ],
        axis=1,
    )

    descriptor_matrix = cleaned_df[
        SELECTED_DESCRIPTORS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    descriptor_matrix = descriptor_matrix.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    descriptor_matrix = descriptor_matrix.mask(
        descriptor_matrix.abs() > MAX_ABS_VALUE,
        np.nan,
    )

    cleaned_df[
        SELECTED_DESCRIPTORS
    ] = descriptor_matrix

    all_missing_descriptor_mask = (
        descriptor_matrix.isna().all(axis=1)
    )

    for _, row in cleaned_df[
        all_missing_descriptor_mask
    ].iterrows():
        removal_rows.append({
            "Original_Row": row["Original_Row"],
            "Reason": "All selected descriptors missing",
            "SMILES": row[SMILES_COL],
        })

    cleaned_df = cleaned_df[
        ~all_missing_descriptor_mask
    ].copy().reset_index(drop=True)

    cleaned_df["Compound_ID"] = [
        f"CMPD_{index:05d}"
        for index in range(1, len(cleaned_df) + 1)
    ]

    cleaned_df.to_csv(
        CLEANED_DATA_PATH,
        index=False,
    )

    pd.DataFrame(removal_rows).to_csv(
        REMOVAL_LOG_PATH,
        index=False,
    )

    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned class counts:")
    print(
        cleaned_df[TARGET_COL]
        .value_counts()
        .sort_index()
    )
    print(
        "Unique scaffolds:",
        cleaned_df["Scaffold"].nunique(),
    )
    print(
        "Removed rows:",
        len(removal_rows),
    )

    return cleaned_df


def load_or_build_cleaned_dataset():
    if os.path.exists(CLEANED_DATA_PATH):
        cleaned_df = pd.read_csv(
            CLEANED_DATA_PATH
        )
        print(
            "Loaded existing cleaned dataset:",
            cleaned_df.shape,
        )
        return cleaned_df

    return build_cleaned_dataset()

# -------------------------
# 8) Create new locked scaffold test
# -------------------------
def create_locked_scaffold_split(cleaned_df):
    if os.path.exists(SPLIT_PATH):
        split_df = pd.read_csv(SPLIT_PATH)

        if set(split_df["Compound_ID"]) != set(
            cleaned_df["Compound_ID"]
        ):
            raise RuntimeError(
                "Existing split file does not match the "
                "current cleaned dataset."
            )

        merged_df = cleaned_df.merge(
            split_df[
                ["Compound_ID", "Partition"]
            ],
            on="Compound_ID",
            how="left",
            validate="one_to_one",
        )

        print(
            "Loaded existing locked scaffold split."
        )
        return merged_df

    labels = cleaned_df[TARGET_COL].to_numpy()
    groups = cleaned_df["Scaffold"].to_numpy()
    overall_prevalence = labels.mean()

    best_candidate = None
    candidate_rows = []

    print(
        "\nSearching candidate scaffold splits ..."
    )
    log_event(
        "scaffold_split_search",
        "started",
        f"candidates={N_SPLIT_CANDIDATES}",
    )

    candidate_seed_iterator = tqdm(
        range(
            RANDOM_STATE,
            RANDOM_STATE + N_SPLIT_CANDIDATES,
        ),
        desc="Scaffold split candidates",
        unit="candidate",
    )

    for candidate_seed in candidate_seed_iterator:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=candidate_seed,
        )

        development_indices, test_indices = next(
            splitter.split(
                cleaned_df,
                labels,
                groups,
            )
        )

        development_labels = labels[
            development_indices
        ]
        test_labels = labels[
            test_indices
        ]

        if (
            len(np.unique(development_labels)) < 2
            or len(np.unique(test_labels)) < 2
        ):
            continue

        actual_test_fraction = (
            len(test_indices) / len(cleaned_df)
        )
        test_prevalence = test_labels.mean()
        development_prevalence = (
            development_labels.mean()
        )

        size_error = abs(
            actual_test_fraction - TEST_SIZE
        )
        prevalence_error = abs(
            test_prevalence - overall_prevalence
        )
        development_prevalence_error = abs(
            development_prevalence
            - overall_prevalence
        )

        objective = (
            2.0 * size_error
            + prevalence_error
            + development_prevalence_error
        )

        candidate_rows.append({
            "Seed": candidate_seed,
            "Objective": objective,
            "Test_Fraction": actual_test_fraction,
            "Test_Prevalence": test_prevalence,
            "Development_Prevalence": (
                development_prevalence
            ),
            "Development_N": len(
                development_indices
            ),
            "Test_N": len(test_indices),
        })

        if (
            best_candidate is None
            or objective < best_candidate[
                "objective"
            ]
        ):
            best_candidate = {
                "objective": objective,
                "seed": candidate_seed,
                "development_indices": (
                    development_indices
                ),
                "test_indices": test_indices,
            }

    if best_candidate is None:
        raise RuntimeError(
            "Unable to create a valid scaffold split."
        )

    partition = np.full(
        len(cleaned_df),
        "development",
        dtype=object,
    )
    partition[
        best_candidate["test_indices"]
    ] = "locked_test"

    split_df = cleaned_df[
        [
            "Compound_ID",
            "Canonical_SMILES",
            "Scaffold",
            TARGET_COL,
        ]
    ].copy()

    split_df["Partition"] = partition
    split_df["Split_Seed"] = best_candidate[
        "seed"
    ]

    split_df.to_csv(
        SPLIT_PATH,
        index=False,
    )

    pd.DataFrame(candidate_rows).sort_values(
        "Objective"
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "scaffold_split_candidate_search.csv",
        ),
        index=False,
    )

    print("\nCreated and locked scaffold split.")
    print(
        split_df["Partition"].value_counts()
    )
    print(
        "\nClass counts by partition:"
    )
    print(
        pd.crosstab(
            split_df["Partition"],
            split_df[TARGET_COL],
        )
    )

    development_scaffolds = set(
        split_df.loc[
            split_df["Partition"]
            == "development",
            "Scaffold",
        ]
    )
    test_scaffolds = set(
        split_df.loc[
            split_df["Partition"]
            == "locked_test",
            "Scaffold",
        ]
    )

    overlap = (
        development_scaffolds
        & test_scaffolds
    )

    if overlap:
        raise RuntimeError(
            "Scaffold leakage detected between "
            "development and locked test."
        )

    return cleaned_df.merge(
        split_df[
            ["Compound_ID", "Partition"]
        ],
        on="Compound_ID",
        how="left",
        validate="one_to_one",
    )

# -------------------------
# 9) Development-only descriptor filtering
# -------------------------
def determine_development_features(
    development_df,
):
    development_descriptors = (
        development_df[
            SELECTED_DESCRIPTORS
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    all_nan_columns = (
        development_descriptors.columns[
            development_descriptors
            .isna()
            .all()
        ].tolist()
    )

    nonempty_descriptors = (
        development_descriptors.drop(
            columns=all_nan_columns,
            errors="ignore",
        )
    )

    constant_columns = (
        nonempty_descriptors.columns[
            nonempty_descriptors
            .nunique(dropna=True)
            <= 1
        ].tolist()
    )

    usable_features = [
        column
        for column in SELECTED_DESCRIPTORS
        if column
        not in set(
            all_nan_columns
            + constant_columns
        )
    ]

    if not usable_features:
        raise RuntimeError(
            "No usable development descriptors."
        )

    return (
        usable_features,
        all_nan_columns,
        constant_columns,
    )

# -------------------------
# 10) Chunk 1: algorithm-family comparison
# -------------------------
def compare_algorithms(
    X_development,
    y_development,
    groups_development,
):
    """
    Compare classical QSAR and tree models using identical outer
    scaffold-grouped folds, descriptors, labels, and threshold.
    """
    algorithms = [
        "LogisticRegression",
        "SVM",
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
    ]

    group_cv = make_group_cv(
        ALGORITHM_CV_FOLDS,
        RANDOM_STATE,
    )

    splits = list(
        group_cv.split(
            X_development,
            y_development,
            groups_development,
        )
    )

    rate_metric_names = [
        "Accuracy",
        "BalancedAccuracy",
        "Precision_toxic",
        "Recall_toxic",
        "Specificity_non_toxic",
        "F1_toxic",
        "MCC",
        "ROC_AUC",
        "PR_AUC",
        "Brier",
        "GO_Precision",
        "GO_Recall",
        "NO_GO_Precision",
        "NO_GO_Recall",
    ]

    fold_result_rows = []
    summary_rows = []
    oof_prediction_rows = []

    print(
        "\nCHUNK 1: Comparing classical QSAR and "
        "tree-based algorithms using identical "
        "scaffold-grouped CV folds ..."
    )
    log_event(
        "chunk1_algorithm_comparison",
        "started",
        f"{len(algorithms)} algorithms; {len(splits)} folds",
    )

    algorithm_progress = tqdm(
        algorithms,
        desc="Model families",
        unit="model",
        position=0,
    )

    for algorithm_name in algorithm_progress:
        algorithm_progress.set_postfix_str(algorithm_name)
        algorithm_fold_rows = []

        oof_probability = np.full(
            len(y_development),
            np.nan,
            dtype=float,
        )
        oof_prediction = np.full(
            len(y_development),
            -1,
            dtype=int,
        )
        oof_fold = np.full(
            len(y_development),
            -1,
            dtype=int,
        )

        fold_progress = tqdm(
            list(enumerate(splits, start=1)),
            desc=f"{algorithm_name} folds",
            unit="fold",
            leave=False,
            position=1,
        )

        for fold_number, (
            training_indices,
            validation_indices,
        ) in fold_progress:
            X_fold_train = X_development.iloc[
                training_indices
            ]
            X_fold_validation = X_development.iloc[
                validation_indices
            ]
            y_fold_train = y_development.iloc[
                training_indices
            ]
            y_fold_validation = y_development.iloc[
                validation_indices
            ]

            positive_weight = calculate_positive_weight(
                y_fold_train
            )

            pipeline = make_algorithm_pipeline(
                algorithm_name,
                positive_weight,
                RANDOM_STATE + fold_number,
            )

            fit_kwargs = {}

            if algorithm_name == "GradientBoosting":
                sample_weights = np.where(
                    y_fold_train.to_numpy() == 1,
                    positive_weight,
                    1.0,
                )
                fit_kwargs[
                    "model__sample_weight"
                ] = sample_weights

            pipeline.fit(
                X_fold_train,
                y_fold_train,
                **fit_kwargs,
            )

            probability = pipeline.predict_proba(
                X_fold_validation
            )[:, 1]

            prediction = (
                probability >= BASELINE_THRESHOLD
            ).astype(int)

            fold_metrics = safe_binary_metrics(
                y_fold_validation,
                prediction,
                probability,
            )
            fold_metrics.update(
                go_no_go_metrics(
                    y_fold_validation,
                    prediction,
                )
            )

            fold_row = {
                "Algorithm": algorithm_name,
                "Fold": fold_number,
                "Threshold": BASELINE_THRESHOLD,
                "Training_N": int(
                    len(training_indices)
                ),
                "Validation_N": int(
                    len(validation_indices)
                ),
                **fold_metrics,
            }

            fold_result_rows.append(fold_row)
            algorithm_fold_rows.append(fold_row)

            oof_probability[
                validation_indices
            ] = probability
            oof_prediction[
                validation_indices
            ] = prediction
            oof_fold[
                validation_indices
            ] = fold_number

            fold_progress.set_postfix(
                PR_AUC=f"{fold_metrics['PR_AUC']:.3f}",
                F1=f"{fold_metrics['F1_toxic']:.3f}",
            )

        if np.isnan(oof_probability).any():
            raise RuntimeError(
                f"Incomplete OOF probabilities for "
                f"{algorithm_name}."
            )

        if (oof_prediction < 0).any():
            raise RuntimeError(
                f"Incomplete OOF predictions for "
                f"{algorithm_name}."
            )

        pooled_metrics = safe_binary_metrics(
            y_development,
            oof_prediction,
            oof_probability,
        )
        pooled_metrics.update(
            go_no_go_metrics(
                y_development,
                oof_prediction,
            )
        )

        algorithm_fold_df = pd.DataFrame(
            algorithm_fold_rows
        )

        summary_row = {
            "Algorithm": algorithm_name,
            "Threshold": BASELINE_THRESHOLD,
            "CV_Folds": ALGORITHM_CV_FOLDS,
            "Validation": (
                "StratifiedGroupKFold grouped by scaffold"
            ),
        }

        for metric_name in rate_metric_names:
            summary_row[
                f"Mean_CV_{metric_name}"
            ] = float(
                algorithm_fold_df[
                    metric_name
                ].mean()
            )
            summary_row[
                f"Std_CV_{metric_name}"
            ] = float(
                algorithm_fold_df[
                    metric_name
                ].std(ddof=1)
            )

        for metric_name, metric_value in pooled_metrics.items():
            summary_row[
                f"OOF_{metric_name}"
            ] = (
                None
                if pd.isna(metric_value)
                else float(metric_value)
            )

        summary_rows.append(summary_row)

        for row_index in range(
            len(y_development)
        ):
            oof_prediction_rows.append({
                "Algorithm": algorithm_name,
                "Development_Row": row_index,
                "Fold": int(
                    oof_fold[row_index]
                ),
                "y_true": int(
                    y_development.iloc[row_index]
                ),
                "y_probability": float(
                    oof_probability[row_index]
                ),
                "y_prediction": int(
                    oof_prediction[row_index]
                ),
                "Threshold": BASELINE_THRESHOLD,
            })

        print(
            f"{algorithm_name}: "
            f"mean PR-AUC = "
            f"{summary_row['Mean_CV_PR_AUC']:.6f} "
            f"± "
            f"{summary_row['Std_CV_PR_AUC']:.6f}; "
            f"OOF F1 = "
            f"{summary_row['OOF_F1_toxic']:.6f}; "
            f"OOF MCC = "
            f"{summary_row['OOF_MCC']:.6f}"
        )

    fold_results = pd.DataFrame(
        fold_result_rows
    )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            "Mean_CV_PR_AUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    oof_results = pd.DataFrame(
        oof_prediction_rows
    )

    fold_results.to_csv(
        ALGORITHM_FOLD_RESULTS_PATH,
        index=False,
    )
    summary.to_csv(
        ALGORITHM_RESULTS_PATH,
        index=False,
    )
    oof_results.to_csv(
        ALGORITHM_OOF_PATH,
        index=False,
    )

    selected_algorithm = summary.loc[
        0,
        "Algorithm",
    ]

    display_columns = [
        "Algorithm",
        "Mean_CV_PR_AUC",
        "Std_CV_PR_AUC",
        "OOF_ROC_AUC",
        "OOF_PR_AUC",
        "OOF_Accuracy",
        "OOF_BalancedAccuracy",
        "OOF_Precision_toxic",
        "OOF_Recall_toxic",
        "OOF_Specificity_non_toxic",
        "OOF_F1_toxic",
        "OOF_MCC",
        "OOF_Brier",
    ]

    print("\nChunk 1 full-metric summary:")
    print(summary[display_columns])
    print(
        "\nSelected algorithm family by mean CV PR-AUC:",
        selected_algorithm,
    )

    log_event(
        "chunk1_algorithm_comparison",
        "completed",
        f"selected_algorithm={selected_algorithm}",
    )

    if selected_algorithm != "XGBoost":
        raise RuntimeError(
            "XGBoost was not the highest-performing "
            "algorithm under the expanded classical-QSAR "
            "and tree-model comparison. The remaining "
            "workflow is XGBoost-specific, so automatic "
            "continuation has been stopped. Review Chunk 1 "
            "before deciding whether to revise the final "
            "model family."
        )

    return summary, selected_algorithm

# -------------------------
# 11) Chunk 2: XGBoost hyperparameter tuning
# -------------------------
def tune_xgboost(
    X_development,
    y_development,
    groups_development,
):
    parameter_space = {
        "n_estimators": [
            300,
            500,
            700,
            900,
        ],
        "max_depth": [
            3,
            4,
            5,
            6,
        ],
        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.07,
        ],
        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "colsample_bytree": [
            0.6,
            0.8,
            1.0,
        ],
        "min_child_weight": [
            1,
            2,
            4,
            6,
            8,
        ],
        "gamma": [
            0,
            0.1,
            0.3,
            0.5,
        ],
        "reg_alpha": [
            0,
            0.01,
            0.1,
            0.5,
            1.0,
        ],
        "reg_lambda": [
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ],
    }

    sampled_parameters = list(
        ParameterSampler(
            parameter_space,
            n_iter=N_PARAM_SAMPLES,
            random_state=RANDOM_STATE,
        )
    )

    tuning_cv = make_group_cv(
        TUNING_CV_FOLDS,
        RANDOM_STATE,
    )

    tuning_splits = list(
        tuning_cv.split(
            X_development,
            y_development,
            groups_development,
        )
    )

    search_rows = []
    best_score = -np.inf
    best_parameters = None

    print(
        "\nCHUNK 2: Tuning XGBoost using "
        "development-only scaffold-grouped CV ..."
    )
    log_event(
        "chunk2_xgboost_tuning",
        "started",
        f"{len(sampled_parameters)} candidates; {len(tuning_splits)} folds",
    )

    candidate_progress = tqdm(
        list(enumerate(sampled_parameters, start=1)),
        desc="XGBoost candidates",
        unit="candidate",
        position=0,
    )

    for candidate_number, parameters in candidate_progress:
        fold_scores = []

        fold_progress = tqdm(
            list(enumerate(tuning_splits, start=1)),
            desc=f"Candidate {candidate_number} folds",
            unit="fold",
            leave=False,
            position=1,
        )

        for fold_number, (
            training_indices,
            validation_indices,
        ) in fold_progress:
            X_fold_train = (
                X_development.iloc[
                    training_indices
                ]
            )
            X_fold_validation = (
                X_development.iloc[
                    validation_indices
                ]
            )
            y_fold_train = (
                y_development.iloc[
                    training_indices
                ]
            )
            y_fold_validation = (
                y_development.iloc[
                    validation_indices
                ]
            )

            fold_positive_weight = (
                calculate_positive_weight(
                    y_fold_train
                )
            )

            pipeline = make_xgb_pipeline(
                parameters,
                fold_positive_weight,
                RANDOM_STATE
                + candidate_number
                + fold_number,
            )

            pipeline.fit(
                X_fold_train,
                y_fold_train,
            )

            probability = (
                pipeline.predict_proba(
                    X_fold_validation
                )[:, 1]
            )

            average_precision = (
                average_precision_score(
                    y_fold_validation,
                    probability,
                )
            )

            fold_scores.append(
                average_precision
            )
            fold_progress.set_postfix(
                PR_AUC=f"{average_precision:.3f}"
            )

        mean_score = float(
            np.mean(fold_scores)
        )
        std_score = float(
            np.std(fold_scores)
        )

        search_row = {
            "Candidate": candidate_number,
            "Mean_CV_AveragePrecision": (
                mean_score
            ),
            "Std_CV_AveragePrecision": (
                std_score
            ),
            **parameters,
        }
        search_rows.append(search_row)

        if mean_score > best_score:
            best_score = mean_score
            best_parameters = (
                parameters.copy()
            )

        candidate_progress.set_postfix(
            best=f"{best_score:.3f}",
            current=f"{mean_score:.3f}",
        )

    search_df = pd.DataFrame(
        search_rows
    ).sort_values(
        "Mean_CV_AveragePrecision",
        ascending=False,
    ).reset_index(drop=True)

    search_df.to_csv(
        TUNING_RESULTS_PATH,
        index=False,
    )

    if best_parameters is None:
        raise RuntimeError(
            "No XGBoost hyperparameters selected."
        )

    print("\nSelected XGBoost parameters:")
    print(
        json.dumps(
            best_parameters,
            indent=2,
        )
    )
    print(
        "Best mean scaffold-CV average "
        f"precision: {best_score:.6f}"
    )

    log_event(
        "chunk2_xgboost_tuning",
        "completed",
        f"best_mean_pr_auc={best_score:.6f}",
    )

    return (
        best_parameters,
        best_score,
        tuning_splits,
    )

# -------------------------
# 12) Select top 60 descriptors
# -------------------------
def select_top_features(
    X_development,
    y_development,
    best_parameters,
):
    print("\nSelecting and freezing top descriptors ...")
    log_event(
        "feature_selection",
        "started",
        f"candidate_features={X_development.shape[1]}",
    )

    development_positive_weight = (
        calculate_positive_weight(
            y_development
        )
    )

    development_pipeline = (
        make_xgb_pipeline(
            best_parameters,
            development_positive_weight,
            RANDOM_STATE,
        )
    )

    development_pipeline.fit(
        X_development,
        y_development,
    )

    model_step = (
        development_pipeline.named_steps[
            "model"
        ]
    )

    feature_importance_df = pd.DataFrame({
        "Feature": X_development.columns,
        "Importance": (
            model_step.feature_importances_
        ),
    }).sort_values(
        "Importance",
        ascending=False,
    ).reset_index(drop=True)

    feature_importance_df[
        "Rank"
    ] = np.arange(
        1,
        len(feature_importance_df) + 1,
    )

    feature_importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "development_feature_importance.csv",
        ),
        index=False,
    )

    selected_features = (
        feature_importance_df.head(
            min(TOP_K, len(feature_importance_df))
        )["Feature"]
        .tolist()
    )

    pd.DataFrame({
        "Rank": np.arange(
            1,
            len(selected_features) + 1,
        ),
        "Feature": selected_features,
    }).to_csv(
        SELECTED_FEATURES_PATH,
        index=False,
    )

    print(
        f"\nFrozen descriptor count: "
        f"{len(selected_features)}"
    )
    log_event(
        "feature_selection",
        "completed",
        f"selected_features={len(selected_features)}",
    )

    return (
        selected_features,
        feature_importance_df,
    )

# -------------------------
# 13) Multicollinearity assessment of frozen descriptors
# -------------------------
def calculate_vif_report(
    X_development_selected,
):
    """
    Calculate variance inflation factors using development data only.

    The selected descriptor matrix is median-imputed and standardized.
    Each descriptor is regressed on all other selected descriptors.
    VIF is diagnostic only: descriptors are not automatically removed,
    because removal would constitute a new feature-selection decision.
    """
    feature_names = list(
        X_development_selected.columns
    )

    if len(feature_names) == 0:
        raise RuntimeError(
            "No selected descriptors are available for VIF."
        )

    log_event(
        "vif_assessment",
        "started",
        f"features={len(feature_names)}",
    )

    if len(feature_names) == 1:
        vif_df = pd.DataFrame([{
            "Feature": feature_names[0],
            "R_squared": 0.0,
            "VIF": 1.0,
            "VIF_Status": "Acceptable",
            "Development_N": int(
                len(X_development_selected)
            ),
            "Auxiliary_Predictor_Count": 0,
        }])
        vif_df.to_csv(
            VIF_RESULTS_PATH,
            index=False,
        )
        log_event(
            "vif_assessment",
            "completed",
            "single_feature",
        )
        return vif_df

    vif_imputer = SimpleImputer(
        strategy="median"
    )
    vif_scaler = StandardScaler()

    X_vif = vif_imputer.fit_transform(
        X_development_selected
    )
    X_vif = vif_scaler.fit_transform(
        X_vif
    )

    development_n = int(X_vif.shape[0])
    auxiliary_predictor_count = int(
        X_vif.shape[1] - 1
    )

    if development_n <= auxiliary_predictor_count + 1:
        print(
            "\nWARNING: The development sample count is not "
            "larger than the number of predictors in each VIF "
            "auxiliary regression. VIF estimates may therefore "
            "be unstable or infinite."
        )

    vif_rows = []

    vif_progress = tqdm(
        list(enumerate(feature_names)),
        desc="Calculating VIF",
        unit="feature",
    )

    for feature_index, feature_name in vif_progress:
        target_feature = X_vif[
            :,
            feature_index,
        ]
        remaining_features = np.delete(
            X_vif,
            feature_index,
            axis=1,
        )

        try:
            auxiliary_model = LinearRegression()
            auxiliary_model.fit(
                remaining_features,
                target_feature,
            )
            r_squared = float(
                auxiliary_model.score(
                    remaining_features,
                    target_feature,
                )
            )
        except Exception:
            r_squared = np.nan

        if pd.isna(r_squared):
            vif_value = np.nan
            vif_status = "Not estimable"
        else:
            r_squared = min(
                max(r_squared, 0.0),
                1.0,
            )

            if r_squared >= 1.0 - 1e-12:
                vif_value = np.inf
            else:
                vif_value = float(
                    1.0 / (1.0 - r_squared)
                )

            if (
                np.isinf(vif_value)
                or vif_value >= VIF_SEVERE_THRESHOLD
            ):
                vif_status = "Severe multicollinearity"
            elif vif_value >= VIF_WARNING_THRESHOLD:
                vif_status = "Potential multicollinearity"
            else:
                vif_status = "Acceptable"

        vif_rows.append({
            "Feature": feature_name,
            "R_squared": r_squared,
            "VIF": vif_value,
            "VIF_Status": vif_status,
            "Development_N": development_n,
            "Auxiliary_Predictor_Count": (
                auxiliary_predictor_count
            ),
        })

        if pd.notna(vif_value) and np.isfinite(vif_value):
            vif_progress.set_postfix(
                VIF=f"{vif_value:.2f}"
            )
        else:
            vif_progress.set_postfix_str(vif_status)

    vif_df = (
        pd.DataFrame(vif_rows)
        .sort_values(
            "VIF",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    vif_df.to_csv(
        VIF_RESULTS_PATH,
        index=False,
    )

    numeric_vif = pd.to_numeric(
        vif_df["VIF"],
        errors="coerce",
    )

    print("\nMULTICOLLINEARITY ASSESSMENT")
    print(
        "VIF was calculated on the frozen development "
        "descriptors after median imputation and scaling."
    )
    print(
        f"Features with VIF >= {VIF_WARNING_THRESHOLD}: ",
        int(
            (
                numeric_vif
                >= VIF_WARNING_THRESHOLD
            ).sum()
        ),
    )
    print(
        f"Features with VIF >= {VIF_SEVERE_THRESHOLD}: ",
        int(
            (
                numeric_vif
                >= VIF_SEVERE_THRESHOLD
            ).sum()
        ),
    )
    print(
        "Infinite VIF values:",
        int(np.isinf(numeric_vif).sum()),
    )
    print(vif_df.head(20))

    log_event(
        "vif_assessment",
        "completed",
        (
            f"warning_or_higher="
            f"{int((numeric_vif >= VIF_WARNING_THRESHOLD).sum())}; "
            f"severe_or_higher="
            f"{int((numeric_vif >= VIF_SEVERE_THRESHOLD).sum())}"
        ),
    )

    return vif_df


# -------------------------
# 14) Development OOF predictions and threshold freeze
# -------------------------
def generate_development_oof_predictions(
    X_development_top,
    y_development,
    groups_development,
    best_parameters,
):
    oof_probability = np.full(
        len(X_development_top),
        np.nan,
        dtype=float,
    )
    oof_fold = np.full(
        len(X_development_top),
        -1,
        dtype=int,
    )

    oof_cv = make_group_cv(
        TUNING_CV_FOLDS,
        RANDOM_STATE + 1000,
    )

    oof_splits = list(
        oof_cv.split(
            X_development_top,
            y_development,
            groups_development,
        )
    )

    print(
        "\nGenerating weighted XGBoost development OOF predictions ..."
    )
    log_event(
        "weighted_oof_predictions",
        "started",
        f"folds={len(oof_splits)}",
    )

    fold_progress = tqdm(
        list(enumerate(oof_splits, start=1)),
        desc="Weighted XGBoost OOF",
        unit="fold",
    )

    for fold_number, (
        training_indices,
        validation_indices,
    ) in fold_progress:
        X_fold_train = (
            X_development_top.iloc[
                training_indices
            ]
        )
        X_fold_validation = (
            X_development_top.iloc[
                validation_indices
            ]
        )
        y_fold_train = (
            y_development.iloc[
                training_indices
            ]
        )

        fold_positive_weight = (
            calculate_positive_weight(
                y_fold_train
            )
        )

        pipeline = make_xgb_pipeline(
            best_parameters,
            fold_positive_weight,
            RANDOM_STATE + 1000 + fold_number,
        )

        pipeline.fit(
            X_fold_train,
            y_fold_train,
        )

        fold_probability = pipeline.predict_proba(
            X_fold_validation
        )[:, 1]

        oof_probability[
            validation_indices
        ] = fold_probability

        oof_fold[
            validation_indices
        ] = fold_number

        fold_progress.set_postfix(
            mean_prob=f"{np.mean(fold_probability):.3f}"
        )

    if np.isnan(oof_probability).any():
        raise RuntimeError(
            "Incomplete development OOF predictions."
        )

    selected_threshold, threshold_table = (
        choose_threshold(
            y_development.to_numpy(),
            oof_probability,
            THRESHOLD_GRID,
            MIN_TOXIC_PRECISION,
        )
    )

    threshold_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "development_threshold_search.csv",
        ),
        index=False,
    )

    oof_prediction = (
        oof_probability
        >= selected_threshold
    ).astype(int)

    oof_metrics = safe_binary_metrics(
        y_development,
        oof_prediction,
        oof_probability,
    )
    oof_metrics.update(
        go_no_go_metrics(
            y_development,
            oof_prediction,
        )
    )

    pd.DataFrame(
        [oof_metrics]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "development_oof_performance.csv",
        ),
        index=False,
    )

    oof_df = pd.DataFrame({
        "Development_Row": (
            np.arange(
                len(y_development)
            )
        ),
        "Fold": oof_fold,
        "y_true": (
            y_development.to_numpy()
        ),
        "y_probability": (
            oof_probability
        ),
        "y_prediction": (
            oof_prediction
        ),
    })

    oof_df.to_csv(
        DEVELOPMENT_OOF_PATH,
        index=False,
    )

    print(
        "\nFrozen threshold:",
        f"{selected_threshold:.2f}",
    )
    print(
        "Development OOF performance:"
    )
    print(pd.DataFrame([oof_metrics]))

    log_event(
        "weighted_oof_predictions",
        "completed",
        (
            f"threshold={selected_threshold:.2f}; "
            f"PR_AUC={oof_metrics['PR_AUC']:.6f}; "
            f"Recall={oof_metrics['Recall_toxic']:.6f}; "
            f"F1={oof_metrics['F1_toxic']:.6f}"
        ),
    )

    return (
        selected_threshold,
        oof_metrics,
        oof_probability,
    )

# -------------------------
# 14) Development-only imbalance-method comparison
# -------------------------
def compare_imbalance_methods_oof(
    X_development_top,
    y_development,
    groups_development,
    best_parameters,
    selected_threshold,
    weighted_oof_probability,
):
    """
    Compare scale_pos_weight, SMOTE, and ADASYN using development-only,
    scaffold-grouped out-of-fold predictions.

    All methods use the same frozen descriptors, XGBoost
    hyperparameters, scaffold folds, and operating threshold. SMOTE
    and ADASYN are fitted only within each training fold.
    """

    oof_cv = make_group_cv(
        TUNING_CV_FOLDS,
        RANDOM_STATE + 1000,
    )

    oof_splits = list(
        oof_cv.split(
            X_development_top,
            y_development,
            groups_development,
        )
    )

    enabled_methods = []
    if RUN_SMOTE_COMPARISON:
        enabled_methods.append("SMOTE")
    if RUN_ADASYN_COMPARISON:
        enabled_methods.append("ADASYN")

    print(
        "\nComparing scale_pos_weight, SMOTE, and ADASYN "
        "using scaffold-grouped OOF predictions ..."
    )
    log_event(
        "imbalance_oof_comparison",
        "started",
        (
            f"methods={['scale_pos_weight'] + enabled_methods}; "
            f"folds={len(oof_splits)}; threshold={selected_threshold}"
        ),
    )

    method_probabilities = {}
    method_folds = {}

    method_progress = tqdm(
        enabled_methods,
        desc="Resampling methods OOF",
        unit="method",
        position=0,
    )

    for method_name in method_progress:
        method_progress.set_postfix_str(method_name)

        method_probability = np.full(
            len(X_development_top),
            np.nan,
            dtype=float,
        )
        method_fold = np.full(
            len(X_development_top),
            -1,
            dtype=int,
        )

        fold_progress = tqdm(
            list(enumerate(oof_splits, start=1)),
            desc=f"{method_name} XGBoost OOF",
            unit="fold",
            leave=False,
            position=1,
        )

        for fold_number, (
            training_indices,
            validation_indices,
        ) in fold_progress:
            X_fold_train = X_development_top.iloc[
                training_indices
            ]
            X_fold_validation = X_development_top.iloc[
                validation_indices
            ]
            y_fold_train = y_development.iloc[
                training_indices
            ]

            training_scaffolds = set(
                groups_development.iloc[
                    training_indices
                ]
            )
            validation_scaffolds = set(
                groups_development.iloc[
                    validation_indices
                ]
            )

            if training_scaffolds & validation_scaffolds:
                raise RuntimeError(
                    f"Scaffold leakage detected in fold {fold_number}."
                )

            minority_sample_count = int(
                (y_fold_train == 1).sum()
            )

            if method_name == "SMOTE":
                method_pipeline = make_smote_xgb_pipeline(
                    params=best_parameters,
                    minority_sample_count=minority_sample_count,
                    random_state=(
                        RANDOM_STATE
                        + 2000
                        + fold_number
                    ),
                )
            elif method_name == "ADASYN":
                method_pipeline = make_adasyn_xgb_pipeline(
                    params=best_parameters,
                    minority_sample_count=minority_sample_count,
                    random_state=(
                        RANDOM_STATE
                        + 3000
                        + fold_number
                    ),
                )
            else:
                raise ValueError(
                    f"Unknown imbalance method: {method_name}"
                )

            try:
                method_pipeline.fit(
                    X_fold_train,
                    y_fold_train,
                )
            except RuntimeError as error:
                if method_name == "ADASYN":
                    raise RuntimeError(
                        "ADASYN failed in scaffold fold "
                        f"{fold_number}: {error}. This can occur when "
                        "the local minority/majority neighborhood does not "
                        "permit adaptive sample generation."
                    ) from error
                raise

            fold_probability = method_pipeline.predict_proba(
                X_fold_validation
            )[:, 1]

            method_probability[
                validation_indices
            ] = fold_probability
            method_fold[
                validation_indices
            ] = fold_number

            fold_progress.set_postfix(
                toxic_train=minority_sample_count,
                mean_prob=f"{np.mean(fold_probability):.3f}",
            )

        if np.isnan(method_probability).any():
            raise RuntimeError(
                f"Incomplete {method_name} OOF predictions."
            )

        method_probabilities[method_name] = method_probability
        method_folds[method_name] = method_fold

    weighted_prediction = (
        weighted_oof_probability
        >= selected_threshold
    ).astype(int)

    weighted_metrics = safe_binary_metrics(
        y_development,
        weighted_prediction,
        weighted_oof_probability,
    )
    weighted_metrics.update(
        go_no_go_metrics(
            y_development,
            weighted_prediction,
        )
    )

    comparison_rows = [{
        "Method": "scale_pos_weight",
        "Threshold": selected_threshold,
        **weighted_metrics,
    }]

    prediction_df = pd.DataFrame({
        "Development_Row": np.arange(
            len(y_development)
        ),
        "y_true": y_development.to_numpy(),
        "weighted_probability": weighted_oof_probability,
        "weighted_prediction": weighted_prediction,
        "Threshold": selected_threshold,
    })

    metric_log_parts = [
        (
            f"weighted_recall={weighted_metrics['Recall_toxic']:.6f}; "
            f"weighted_f1={weighted_metrics['F1_toxic']:.6f}; "
            f"weighted_pr_auc={weighted_metrics['PR_AUC']:.6f}"
        )
    ]

    for method_name in enabled_methods:
        method_probability = method_probabilities[
            method_name
        ]
        method_prediction = (
            method_probability >= selected_threshold
        ).astype(int)

        method_metrics = safe_binary_metrics(
            y_development,
            method_prediction,
            method_probability,
        )
        method_metrics.update(
            go_no_go_metrics(
                y_development,
                method_prediction,
            )
        )

        comparison_rows.append({
            "Method": method_name,
            "Threshold": selected_threshold,
            **method_metrics,
        })

        method_key = method_name.lower()
        prediction_df[
            f"{method_key}_fold"
        ] = method_folds[method_name]
        prediction_df[
            f"{method_key}_probability"
        ] = method_probability
        prediction_df[
            f"{method_key}_prediction"
        ] = method_prediction

        metric_log_parts.append(
            (
                f"{method_key}_recall="
                f"{method_metrics['Recall_toxic']:.6f}; "
                f"{method_key}_f1="
                f"{method_metrics['F1_toxic']:.6f}; "
                f"{method_key}_pr_auc="
                f"{method_metrics['PR_AUC']:.6f}"
            )
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        IMBALANCE_OOF_RESULTS_PATH,
        index=False,
    )

    prediction_df.to_csv(
        IMBALANCE_OOF_PREDICTIONS_PATH,
        index=False,
    )

    print("\nDevelopment OOF imbalance comparison:")
    print(
        comparison_df[
            [
                "Method",
                "Recall_toxic",
                "F1_toxic",
                "PR_AUC",
                "Precision_toxic",
                "BalancedAccuracy",
                "MCC",
                "ROC_AUC",
                "Accuracy",
                "Threshold",
            ]
        ]
    )

    log_event(
        "imbalance_oof_comparison",
        "completed",
        "; ".join(metric_log_parts),
    )

    return comparison_df

# -------------------------
# 15) Freeze development configuration
# -------------------------
def save_frozen_configuration(
    cleaned_df,
    development_df,
    locked_test_df,
    usable_features,
    all_nan_features,
    constant_features,
    selected_algorithm,
    algorithm_summary,
    best_parameters,
    best_tuning_score,
    selected_features,
    vif_report,
    selected_threshold,
    development_oof_metrics,
):
    vif_values = pd.to_numeric(
        vif_report["VIF"],
        errors="coerce",
    )
    finite_vif_values = vif_values[
        np.isfinite(vif_values)
    ]
    maximum_finite_vif = (
        None
        if finite_vif_values.empty
        else float(finite_vif_values.max())
    )

    configuration = {
        "workflow": (
            "Cleaned scaffold-locked toxicity workflow with "
            "classical QSAR baselines, VIF, SMOTE, and ADASYN comparisons"
        ),
        "run_mode_used_for_freezing": (
            "development"
        ),
        "raw_data_filename": DATA_FILENAME,
        "cleaned_dataset_path": (
            CLEANED_DATA_PATH
        ),
        "locked_split_path": SPLIT_PATH,
        "run_log_path": RUN_LOG_PATH,
        "structured_event_log_path": RUN_EVENT_LOG_PATH,
        "cleaned_compound_count": int(
            len(cleaned_df)
        ),
        "development_compound_count": int(
            len(development_df)
        ),
        "locked_test_compound_count": int(
            len(locked_test_df)
        ),
        "development_class_counts": {
            str(key): int(value)
            for key, value in (
                development_df[TARGET_COL]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "locked_test_class_counts": {
            str(key): int(value)
            for key, value in (
                locked_test_df[TARGET_COL]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "development_scaffold_count": int(
            development_df[
                "Scaffold"
            ].nunique()
        ),
        "locked_test_scaffold_count": int(
            locked_test_df[
                "Scaffold"
            ].nunique()
        ),
        "initial_descriptor_count": int(
            len(SELECTED_DESCRIPTORS)
        ),
        "usable_descriptor_count": int(
            len(usable_features)
        ),
        "development_all_nan_descriptors": (
            all_nan_features
        ),
        "development_constant_descriptors": (
            constant_features
        ),
        "selected_algorithm": (
            selected_algorithm
        ),
        "algorithm_selection_metric": (
            "Mean scaffold-grouped CV PR-AUC using "
            "identical folds for all model families"
        ),
        "algorithm_comparison_threshold": (
            BASELINE_THRESHOLD
        ),
        "algorithm_comparison_folds": (
            ALGORITHM_CV_FOLDS
        ),
        "algorithm_families_compared": [
            "LogisticRegression",
            "SVM",
            "RandomForest",
            "GradientBoosting",
            "XGBoost",
        ],
        "algorithm_comparison": (
            algorithm_summary.to_dict(
                orient="records"
            )
        ),
        "xgboost_tuning_metric": (
            "Mean scaffold-grouped CV "
            "average precision"
        ),
        "xgboost_tuning_candidates": (
            N_PARAM_SAMPLES
        ),
        "xgboost_tuning_folds": (
            TUNING_CV_FOLDS
        ),
        "best_tuning_score": float(
            best_tuning_score
        ),
        "fixed_params": best_parameters,
        "selected_feature_count": int(
            len(selected_features)
        ),
        "selected_features": (
            selected_features
        ),
        "multicollinearity_assessment": {
            "method": (
                "Variance inflation factor calculated on "
                "median-imputed and standardized frozen "
                "development descriptors"
            ),
            "results_path": VIF_RESULTS_PATH,
            "importance_with_vif_path": (
                IMPORTANCE_WITH_VIF_PATH
            ),
            "warning_threshold": (
                VIF_WARNING_THRESHOLD
            ),
            "severe_threshold": (
                VIF_SEVERE_THRESHOLD
            ),
            "maximum_finite_vif": (
                maximum_finite_vif
            ),
            "infinite_vif_count": int(
                np.isinf(vif_values).sum()
            ),
            "not_estimable_count": int(
                vif_values.isna().sum()
            ),
            "features_vif_ge_warning": int(
                (
                    vif_values
                    >= VIF_WARNING_THRESHOLD
                ).sum()
            ),
            "features_vif_ge_severe": int(
                (
                    vif_values
                    >= VIF_SEVERE_THRESHOLD
                ).sum()
            ),
            "interpretation": (
                "High-VIF descriptors are retained for the "
                "frozen tree model, but individual importance "
                "values should be interpreted as potentially "
                "shared among correlated descriptors."
            ),
        },
        "threshold_selection_source": (
            "Development-only scaffold-grouped "
            "out-of-fold predictions from the "
            "scale_pos_weight model"
        ),
        "minimum_toxic_precision_constraint": (
            MIN_TOXIC_PRECISION
        ),
        "fixed_threshold": float(
            selected_threshold
        ),
        "development_oof_metrics": {
            key: (
                None
                if pd.isna(value)
                else float(value)
            )
            for key, value
            in development_oof_metrics.items()
        },
        "imbalance_methods": {
            "primary": "scale_pos_weight",
            "comparisons": [
                method_name
                for method_name, enabled in [
                    ("SMOTE", RUN_SMOTE_COMPARISON),
                    ("ADASYN", RUN_ADASYN_COMPARISON),
                ]
                if enabled
            ],
            "smote_k_neighbors": (
                SMOTE_K_NEIGHBORS
            ),
            "smote_sampling_strategy": (
                SMOTE_SAMPLING_STRATEGY
            ),
            "adasyn_n_neighbors": (
                ADASYN_N_NEIGHBORS
            ),
            "adasyn_sampling_strategy": (
                ADASYN_SAMPLING_STRATEGY
            ),
            "comparison_threshold": float(
                selected_threshold
            ),
            "comparison_note": (
                "SMOTE and ADASYN use the same frozen features, "
                "hyperparameters, scaffold folds, and threshold."
            ),
        },
        "applicability_domain": {
            "descriptor_space": (
                "Frozen top-60 descriptors"
            ),
            "imputation": "Development median",
            "scaling": (
                "Development StandardScaler"
            ),
            "neighbors": AD_K,
            "cutoff_percentile": (
                AD_PERCENTILE_CUTOFF
            ),
        },
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "locked_test_evaluated": False,
    }

    with open(
        FROZEN_CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            configuration,
            handle,
            indent=2,
        )

    print(
        "\nFrozen configuration saved to:"
    )
    print(FROZEN_CONFIG_PATH)
    log_event(
        "freeze_configuration",
        "completed",
        FROZEN_CONFIG_PATH,
    )

# -------------------------
# 16) Development mode
# -------------------------
def run_development(
    cleaned_with_split_df,
):
    log_event("development_mode", "started")

    development_df = (
        cleaned_with_split_df.loc[
            cleaned_with_split_df[
                "Partition"
            ] == "development"
        ].copy().reset_index(drop=True)
    )

    # Locked test labels are not used below.
    locked_test_df = (
        cleaned_with_split_df.loc[
            cleaned_with_split_df[
                "Partition"
            ] == "locked_test"
        ].copy().reset_index(drop=True)
    )

    (
        usable_features,
        all_nan_features,
        constant_features,
    ) = determine_development_features(
        development_df
    )

    X_development = development_df[
        usable_features
    ].copy()
    y_development = development_df[
        TARGET_COL
    ].copy()
    groups_development = development_df[
        "Scaffold"
    ].copy()

    algorithm_summary, selected_algorithm = (
        compare_algorithms(
            X_development,
            y_development,
            groups_development,
        )
    )

    (
        best_parameters,
        best_tuning_score,
        _,
    ) = tune_xgboost(
        X_development,
        y_development,
        groups_development,
    )

    (
        selected_features,
        feature_importance_df,
    ) = select_top_features(
        X_development,
        y_development,
        best_parameters,
    )

    X_development_top = (
        X_development[
            selected_features
        ].copy()
    )

    vif_report = calculate_vif_report(
        X_development_top
    )

    importance_with_vif_df = (
        feature_importance_df.loc[
            feature_importance_df[
                "Feature"
            ].isin(selected_features)
        ]
        .merge(
            vif_report[
                [
                    "Feature",
                    "R_squared",
                    "VIF",
                    "VIF_Status",
                ]
            ],
            on="Feature",
            how="left",
            validate="one_to_one",
        )
        .sort_values("Rank")
        .reset_index(drop=True)
    )

    importance_with_vif_df.to_csv(
        IMPORTANCE_WITH_VIF_PATH,
        index=False,
    )

    (
        selected_threshold,
        development_oof_metrics,
        weighted_oof_probability,
    ) = generate_development_oof_predictions(
        X_development_top,
        y_development,
        groups_development,
        best_parameters,
    )

    if RUN_SMOTE_COMPARISON or RUN_ADASYN_COMPARISON:
        compare_imbalance_methods_oof(
            X_development_top=(
                X_development_top
            ),
            y_development=y_development,
            groups_development=(
                groups_development
            ),
            best_parameters=best_parameters,
            selected_threshold=(
                selected_threshold
            ),
            weighted_oof_probability=(
                weighted_oof_probability
            ),
        )

    save_frozen_configuration(
        cleaned_with_split_df,
        development_df,
        locked_test_df,
        usable_features,
        all_nan_features,
        constant_features,
        selected_algorithm,
        algorithm_summary,
        best_parameters,
        best_tuning_score,
        selected_features,
        vif_report,
        selected_threshold,
        development_oof_metrics,
    )

    log_event("development_mode", "completed")

    print(
        "\nDEVELOPMENT COMPLETE."
        "\nThe locked scaffold test was not evaluated."
        "\nArchive the output directory."
        "\nThen change only RUN_MODE to "
        "'final_test' and rerun."
    )

# -------------------------
# 17) Chunk 3: locked scaffold-test evaluation
# -------------------------
def run_final_test(
    cleaned_with_split_df,
):
    log_event("final_test_mode", "started")
    if not os.path.exists(
        FROZEN_CONFIG_PATH
    ):
        raise FileNotFoundError(
            "Frozen configuration not found. "
            "Run development mode first."
        )

    if (
        os.path.exists(FINAL_FLAG_PATH)
        and not ALLOW_REPEAT_FINAL_TEST
    ):
        raise RuntimeError(
            "The locked test has already been "
            "evaluated. Set ALLOW_REPEAT_FINAL_TEST "
            "to True only for exact reproducibility "
            "checks, never for further model tuning."
        )

    with open(
        FROZEN_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as handle:
        configuration = json.load(handle)

    if (
        configuration["selected_algorithm"]
        != "XGBoost"
    ):
        raise RuntimeError(
            "Frozen selected algorithm is not "
            "XGBoost."
        )

    best_parameters = configuration[
        "fixed_params"
    ]
    selected_features = configuration[
        "selected_features"
    ]
    selected_threshold = float(
        configuration["fixed_threshold"]
    )

    final_progress = tqdm(
        total=5,
        desc="Final-test stages",
        unit="stage",
    )

    development_df = (
        cleaned_with_split_df.loc[
            cleaned_with_split_df[
                "Partition"
            ] == "development"
        ].copy().reset_index(drop=True)
    )

    locked_test_df = (
        cleaned_with_split_df.loc[
            cleaned_with_split_df[
                "Partition"
            ] == "locked_test"
        ].copy().reset_index(drop=True)
    )

    X_development = development_df[
        selected_features
    ].copy()
    y_development = development_df[
        TARGET_COL
    ].copy()

    X_locked_test = locked_test_df[
        selected_features
    ].copy()
    y_locked_test = locked_test_df[
        TARGET_COL
    ].copy()

    development_positive_weight = (
        calculate_positive_weight(
            y_development
        )
    )

    final_pipeline = make_xgb_pipeline(
        best_parameters,
        development_positive_weight,
        RANDOM_STATE,
    )

    final_pipeline.fit(
        X_development,
        y_development,
    )

    test_probability = (
        final_pipeline.predict_proba(
            X_locked_test
        )[:, 1]
    )

    test_prediction = (
        test_probability
        >= selected_threshold
    ).astype(int)

    final_metrics = safe_binary_metrics(
        y_locked_test,
        test_prediction,
        test_probability,
    )
    final_metrics.update(
        go_no_go_metrics(
            y_locked_test,
            test_prediction,
        )
    )
    final_metrics[
        "Threshold"
    ] = selected_threshold

    final_progress.update(1)
    final_progress.set_postfix_str("weighted model evaluated")
    log_event(
        "weighted_locked_test",
        "completed",
        (
            f"Recall={final_metrics['Recall_toxic']:.6f}; "
            f"F1={final_metrics['F1_toxic']:.6f}; "
            f"PR_AUC={final_metrics['PR_AUC']:.6f}"
        ),
    )

    # -----------------------------------------------------
    # SMOTE and ADASYN comparisons on the unchanged locked test
    # -----------------------------------------------------
    smote_metrics = None
    smote_test_probability = None
    smote_test_prediction = None
    smote_pipeline = None

    adasyn_metrics = None
    adasyn_test_probability = None
    adasyn_test_prediction = None
    adasyn_pipeline = None

    imbalance_rows = [{
        "Method": "scale_pos_weight",
        **final_metrics,
    }]

    enabled_resampling_methods = []
    if RUN_SMOTE_COMPARISON:
        enabled_resampling_methods.append("SMOTE")
    if RUN_ADASYN_COMPARISON:
        enabled_resampling_methods.append("ADASYN")

    development_toxic_count = int(
        (y_development == 1).sum()
    )

    resampling_progress = tqdm(
        enabled_resampling_methods,
        desc="Locked-test resampling methods",
        unit="method",
        leave=False,
    )

    for method_name in resampling_progress:
        resampling_progress.set_postfix_str(method_name)

        if method_name == "SMOTE":
            method_pipeline = make_smote_xgb_pipeline(
                params=best_parameters,
                minority_sample_count=(
                    development_toxic_count
                ),
                random_state=RANDOM_STATE,
            )
        elif method_name == "ADASYN":
            method_pipeline = make_adasyn_xgb_pipeline(
                params=best_parameters,
                minority_sample_count=(
                    development_toxic_count
                ),
                random_state=RANDOM_STATE + 1,
            )
        else:
            raise ValueError(
                f"Unknown imbalance method: {method_name}"
            )

        try:
            method_pipeline.fit(
                X_development,
                y_development,
            )
        except RuntimeError as error:
            if method_name == "ADASYN":
                raise RuntimeError(
                    "ADASYN failed on the full development set: "
                    f"{error}. This can occur when the local "
                    "minority/majority neighborhood does not permit "
                    "adaptive sample generation."
                ) from error
            raise

        method_probability = (
            method_pipeline.predict_proba(
                X_locked_test
            )[:, 1]
        )

        method_prediction = (
            method_probability
            >= selected_threshold
        ).astype(int)

        method_metrics = safe_binary_metrics(
            y_locked_test,
            method_prediction,
            method_probability,
        )
        method_metrics.update(
            go_no_go_metrics(
                y_locked_test,
                method_prediction,
            )
        )
        method_metrics[
            "Threshold"
        ] = selected_threshold

        imbalance_rows.append({
            "Method": method_name,
            **method_metrics,
        })

        if method_name == "SMOTE":
            smote_pipeline = method_pipeline
            smote_test_probability = method_probability
            smote_test_prediction = method_prediction
            smote_metrics = method_metrics
            joblib.dump(
                smote_pipeline,
                SMOTE_MODEL_PATH,
            )
        else:
            adasyn_pipeline = method_pipeline
            adasyn_test_probability = method_probability
            adasyn_test_prediction = method_prediction
            adasyn_metrics = method_metrics
            joblib.dump(
                adasyn_pipeline,
                ADASYN_MODEL_PATH,
            )

        log_event(
            f"{method_name.lower()}_locked_test",
            "completed",
            (
                f"Recall={method_metrics['Recall_toxic']:.6f}; "
                f"F1={method_metrics['F1_toxic']:.6f}; "
                f"PR_AUC={method_metrics['PR_AUC']:.6f}"
            ),
        )

    imbalance_comparison_df = pd.DataFrame(
        imbalance_rows
    )
    imbalance_comparison_df.to_csv(
        IMBALANCE_TEST_RESULTS_PATH,
        index=False,
    )

    print(
        "\nLocked-test class-imbalance comparison:"
    )
    print(
        imbalance_comparison_df[
            [
                "Method",
                "Recall_toxic",
                "F1_toxic",
                "PR_AUC",
                "Precision_toxic",
                "BalancedAccuracy",
                "MCC",
                "ROC_AUC",
                "Accuracy",
                "Threshold",
            ]
        ]
    )

    final_progress.update(1)
    final_progress.set_postfix_str(
        "imbalance comparisons evaluated"
        if enabled_resampling_methods
        else "imbalance comparisons skipped"
    )

    # Save the original weighted-model result file unchanged.
    pd.DataFrame(
        [final_metrics]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "locked_scaffold_test_results.csv",
        ),
        index=False,
    )

    classification_report_df = (
        pd.DataFrame(
            classification_report(
                y_locked_test,
                test_prediction,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
    )

    classification_report_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            (
                "locked_scaffold_test_"
                "classification_report.csv"
            ),
        )
    )

    confusion_df = pd.DataFrame(
        confusion_matrix(
            y_locked_test,
            test_prediction,
            labels=[0, 1],
        ),
        index=[
            "Actual_GO",
            "Actual_NO_GO",
        ],
        columns=[
            "Predicted_GO",
            "Predicted_NO_GO",
        ],
    )

    confusion_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            (
                "locked_scaffold_test_"
                "confusion_matrix.csv"
            ),
        )
    )

    if smote_test_prediction is not None:
        smote_classification_report_df = (
            pd.DataFrame(
                classification_report(
                    y_locked_test,
                    smote_test_prediction,
                    output_dict=True,
                    zero_division=0,
                )
            ).transpose()
        )
        smote_classification_report_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "locked_scaffold_test_smote_classification_report.csv",
            )
        )

        smote_confusion_df = pd.DataFrame(
            confusion_matrix(
                y_locked_test,
                smote_test_prediction,
                labels=[0, 1],
            ),
            index=[
                "Actual_GO",
                "Actual_NO_GO",
            ],
            columns=[
                "Predicted_GO",
                "Predicted_NO_GO",
            ],
        )
        smote_confusion_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "locked_scaffold_test_smote_confusion_matrix.csv",
            )
        )

    if adasyn_test_prediction is not None:
        adasyn_classification_report_df = (
            pd.DataFrame(
                classification_report(
                    y_locked_test,
                    adasyn_test_prediction,
                    output_dict=True,
                    zero_division=0,
                )
            ).transpose()
        )
        adasyn_classification_report_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "locked_scaffold_test_adasyn_classification_report.csv",
            )
        )

        adasyn_confusion_df = pd.DataFrame(
            confusion_matrix(
                y_locked_test,
                adasyn_test_prediction,
                labels=[0, 1],
            ),
            index=[
                "Actual_GO",
                "Actual_NO_GO",
            ],
            columns=[
                "Predicted_GO",
                "Predicted_NO_GO",
            ],
        )
        adasyn_confusion_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "locked_scaffold_test_adasyn_confusion_matrix.csv",
            )
        )

    prediction_columns = [
        "Compound_ID",
        "Canonical_SMILES",
        "Scaffold",
        TARGET_COL,
    ]

    for column in KEEP_ID_COLS:
        if column in locked_test_df.columns:
            prediction_columns.insert(
                1,
                column,
            )

    prediction_df = locked_test_df[
        prediction_columns
    ].copy()

    prediction_df.rename(
        columns={TARGET_COL: "y_true"},
        inplace=True,
    )
    prediction_df[
        "y_probability_toxic"
    ] = test_probability
    prediction_df[
        "y_prediction"
    ] = test_prediction
    prediction_df["Decision"] = np.where(
        test_prediction == 1,
        "NO-GO",
        "GO",
    )

    if smote_test_probability is not None:
        prediction_df[
            "smote_probability_toxic"
        ] = smote_test_probability
        prediction_df[
            "smote_prediction"
        ] = smote_test_prediction
        prediction_df[
            "smote_decision"
        ] = np.where(
            smote_test_prediction == 1,
            "NO-GO",
            "GO",
        )

    if adasyn_test_probability is not None:
        prediction_df[
            "adasyn_probability_toxic"
        ] = adasyn_test_probability
        prediction_df[
            "adasyn_prediction"
        ] = adasyn_test_prediction
        prediction_df[
            "adasyn_decision"
        ] = np.where(
            adasyn_test_prediction == 1,
            "NO-GO",
            "GO",
        )

    joblib.dump(
        final_pipeline,
        FINAL_MODEL_PATH,
    )

    print(
        "\nCHUNK 3: Locked scaffold-test results "
        "for the original weighted model"
    )
    print(pd.DataFrame([final_metrics]))
    print("\nConfusion matrix:")
    print(confusion_df)

    # -------------------------
    # 18) Chunk 4A: applicability domain
    # -------------------------
    imputer_ad = SimpleImputer(
        strategy="median"
    )
    scaler_ad = StandardScaler()

    X_development_imputed = (
        imputer_ad.fit_transform(
            X_development
        )
    )
    X_test_imputed = (
        imputer_ad.transform(
            X_locked_test
        )
    )

    X_development_scaled = (
        scaler_ad.fit_transform(
            X_development_imputed
        )
    )
    X_test_scaled = (
        scaler_ad.transform(
            X_test_imputed
        )
    )

    train_neighbor_model = (
        NearestNeighbors(
            n_neighbors=AD_K + 1,
            metric="euclidean",
        )
    )
    train_neighbor_model.fit(
        X_development_scaled
    )

    train_distances, _ = (
        train_neighbor_model.kneighbors(
            X_development_scaled
        )
    )

    train_mean_distance = (
        train_distances[
            :,
            1:(AD_K + 1),
        ].mean(axis=1)
    )

    test_neighbor_model = (
        NearestNeighbors(
            n_neighbors=AD_K,
            metric="euclidean",
        )
    )
    test_neighbor_model.fit(
        X_development_scaled
    )

    test_distances, _ = (
        test_neighbor_model.kneighbors(
            X_test_scaled
        )
    )

    test_mean_distance = (
        test_distances.mean(axis=1)
    )

    ad_cutoff = float(
        np.percentile(
            train_mean_distance,
            AD_PERCENTILE_CUTOFF,
        )
    )

    test_inside_ad = (
        test_mean_distance <= ad_cutoff
    )

    prediction_df[
        "mean_kNN_distance"
    ] = test_mean_distance
    prediction_df[
        "inside_applicability_domain"
    ] = test_inside_ad

    prediction_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            (
                "locked_scaffold_test_"
                "predictions_with_ad.csv"
            ),
        ),
        index=False,
    )

    ad_rows = []

    for subset_name, mask in [
        ("Inside_AD", test_inside_ad),
        ("Outside_AD", ~test_inside_ad),
    ]:
        if mask.sum() == 0:
            continue

        subset_metrics = safe_binary_metrics(
            y_locked_test[mask],
            test_prediction[mask],
            test_probability[mask],
        )

        subset_metrics.update(
            go_no_go_metrics(
                y_locked_test[mask],
                test_prediction[mask],
            )
        )

        subset_metrics["Subset"] = (
            subset_name
        )
        subset_metrics["N"] = int(
            mask.sum()
        )
        ad_rows.append(subset_metrics)

    ad_results_df = pd.DataFrame(
        ad_rows
    )
    ad_results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            (
                "locked_scaffold_test_"
                "applicability_domain_results.csv"
            ),
        ),
        index=False,
    )

    pd.DataFrame([{
        "AD_K": AD_K,
        "AD_Percentile_Cutoff": (
            AD_PERCENTILE_CUTOFF
        ),
        "AD_Distance_Cutoff": (
            ad_cutoff
        ),
        "Inside_AD_N": int(
            test_inside_ad.sum()
        ),
        "Outside_AD_N": int(
            (~test_inside_ad).sum()
        ),
    }]).to_csv(
        os.path.join(
            OUTPUT_DIR,
            (
                "applicability_domain_"
                "definition.csv"
            ),
        ),
        index=False,
    )

    print(
        "\nCHUNK 4A: Applicability domain"
    )
    print(
        "AD cutoff:",
        f"{ad_cutoff:.4f}",
    )
    print(
        "Inside AD:",
        int(test_inside_ad.sum()),
        "/",
        len(test_inside_ad),
    )
    print(ad_results_df)

    final_progress.update(1)
    final_progress.set_postfix_str("applicability domain completed")
    log_event(
        "applicability_domain",
        "completed",
        (
            f"cutoff={ad_cutoff:.6f}; "
            f"inside={int(test_inside_ad.sum())}; "
            f"outside={int((~test_inside_ad).sum())}"
        ),
    )

    # -------------------------
    # 19) Chunk 4B: SHAP interpretation
    # -------------------------
    final_imputer = (
        final_pipeline.named_steps[
            "imputer"
        ]
    )
    final_model = (
        final_pipeline.named_steps[
            "model"
        ]
    )

    X_test_for_shap = (
        final_imputer.transform(
            X_locked_test
        )
    )

    explainer = shap.TreeExplainer(
        final_model
    )
    shap_values = explainer.shap_values(
        X_test_for_shap
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Mean_Absolute_SHAP": (
            np.abs(shap_values)
            .mean(axis=0)
        ),
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    if os.path.exists(VIF_RESULTS_PATH):
        vif_report = pd.read_csv(
            VIF_RESULTS_PATH
        )
        shap_importance_df = (
            shap_importance_df.merge(
                vif_report[
                    [
                        "Feature",
                        "VIF",
                        "VIF_Status",
                    ]
                ],
                on="Feature",
                how="left",
                validate="one_to_one",
            )
        )

    shap_importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "locked_test_shap_importance.csv",
        ),
        index=False,
    )

    shap_values_df = pd.DataFrame(
        shap_values,
        columns=selected_features,
    )
    shap_values_df.insert(
        0,
        "Compound_ID",
        locked_test_df[
            "Compound_ID"
        ].to_numpy(),
    )
    shap_values_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "locked_test_shap_values.csv",
        ),
        index=False,
    )

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_for_shap,
        feature_names=selected_features,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "locked_test_shap_bar_top20.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_for_shap,
        feature_names=selected_features,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "locked_test_shap_beeswarm_top20.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\nCHUNK 4B: Top SHAP features")
    print(shap_importance_df.head(20))

    final_progress.update(1)
    final_progress.set_postfix_str("SHAP completed")
    log_event(
        "shap_interpretation",
        "completed",
        f"features={len(selected_features)}",
    )

    configuration[
        "locked_test_evaluated"
    ] = True
    configuration[
        "locked_test_metrics"
    ] = {
        key: (
            None
            if pd.isna(value)
            else float(value)
        )
        for key, value in final_metrics.items()
    }

    if smote_metrics is not None:
        configuration[
            "locked_test_smote_metrics"
        ] = {
            key: (
                None
                if pd.isna(value)
                else float(value)
            )
            for key, value in smote_metrics.items()
        }

    if adasyn_metrics is not None:
        configuration[
            "locked_test_adasyn_metrics"
        ] = {
            key: (
                None
                if pd.isna(value)
                else float(value)
            )
            for key, value in adasyn_metrics.items()
        }

    configuration[
        "ad_distance_cutoff"
    ] = ad_cutoff
    configuration[
        "inside_ad_count"
    ] = int(
        test_inside_ad.sum()
    )
    configuration[
        "outside_ad_count"
    ] = int(
        (~test_inside_ad).sum()
    )

    with open(
        FROZEN_CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            configuration,
            handle,
            indent=2,
        )

    with open(
        FINAL_FLAG_PATH,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "Locked scaffold test evaluated.\n"
        )

    final_progress.update(1)
    final_progress.set_postfix_str("outputs saved")
    final_progress.close()
    log_event("final_test_mode", "completed")

    print(
        "\nFINAL TEST COMPLETE."
        "\nDo not use these results for further "
        "hyperparameter, feature, or threshold tuning."
    )

# -------------------------
# 20) Save software versions
# -------------------------
def save_software_versions():
    version_record = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "xgboost": xgboost.__version__,
        "rdkit": rdBase.rdkitVersion,
        "shap": shap.__version__,
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "software_versions.json",
        ),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            version_record,
            handle,
            indent=2,
        )

    log_event(
        "software_versions",
        "saved",
        json.dumps(version_record),
    )

# -------------------------
# 21) Execute selected mode
# -------------------------
try:
    save_software_versions()

    cleaned_df = load_or_build_cleaned_dataset()
    cleaned_with_split_df = (
        create_locked_scaffold_split(
            cleaned_df
        )
    )

    if RUN_MODE == "development":
        run_development(
            cleaned_with_split_df
        )

    elif RUN_MODE == "final_test":
        run_final_test(
            cleaned_with_split_df
        )

    else:
        raise ValueError(
            "RUN_MODE must be 'development' "
            "or 'final_test'."
        )

    log_event("workflow", "completed", f"RUN_MODE={RUN_MODE}")

except Exception as error:
    log_event(
        "workflow",
        "failed",
        f"{type(error).__name__}: {error}",
    )
    print("\nWORKFLOW FAILED")
    traceback.print_exc()
    raise

finally:
    print("\n" + "=" * 72)
    print("Workflow console log exported to:")
    print(RUN_LOG_PATH)
    print("Structured workflow event log exported to:")
    print(RUN_EVENT_LOG_PATH)
    print("Data-cleaning removal log:")
    print(REMOVAL_LOG_PATH)
    print("Finished (UTC):", utc_timestamp())
    print("=" * 72)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = _ORIGINAL_STDOUT
        sys.stderr = _ORIGINAL_STDERR
        _RUN_LOG_HANDLE.close()

