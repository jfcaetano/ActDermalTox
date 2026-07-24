#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 01:11:42 2026

@author: jfcaetano
"""

# STRUCTURAL ERROR ANALYSIS OF FALSE-NEGATIVE TOXIC COMPOUNDS


from pathlib import Path
import atexit
import hashlib
import importlib.metadata as importlib_metadata
import platform
import sys
import traceback
import warnings
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator

warnings.filterwarnings("default")


# ============================================================
# 1. User settings
# ============================================================

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/ChemEng/"
    "results_scaffold_locked_qsar_vif_smote_adasyn"
)

SELECTED_THRESHOLD = 0.56
TOP_STRUCTURES_TO_DRAW = 24
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

PREDICTION_PATH = (
    OUTPUT_DIR
    / "locked_scaffold_test_predictions_with_ad.csv"
)

CLEANED_DATA_PATH = (
    OUTPUT_DIR
    / "cleaned_standardized_dataset.csv"
)

SPLIT_PATH = (
    OUTPUT_DIR
    / "locked_scaffold_split_assignments.csv"
)

SELECTED_FEATURES_PATH = (
    OUTPUT_DIR
    / "frozen_top60_features.csv"
)


# ============================================================
# 1A. Create a timestamped execution log for Supporting Information
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_START_UTC = datetime.now(timezone.utc)
RUN_TIMESTAMP = RUN_START_UTC.strftime("%Y%m%d_%H%M%S")

LOG_PATH = (
    OUTPUT_DIR
    / f"structural_error_analysis_execution_log_{RUN_TIMESTAMP}.txt"
)


class TeeStream:
    """
    Duplicate terminal output to both the notebook console and a text log.

    The original stream is retained so rerunning the cell does not create
    nested TeeStream objects.
    """

    def __init__(self, original_stream, log_stream):
        self._tee_original = getattr(
            original_stream,
            "_tee_original",
            original_stream,
        )
        self.log_stream = log_stream

    def write(self, message):
        self._tee_original.write(message)
        self.log_stream.write(message)
        self.flush()
        return len(message)

    def flush(self):
        self._tee_original.flush()
        self.log_stream.flush()

    def __getattr__(self, attribute):
        return getattr(self._tee_original, attribute)


LOG_FILE = LOG_PATH.open(
    mode="w",
    encoding="utf-8",
    errors="replace",
    buffering=1,
)

ORIGINAL_STDOUT = getattr(
    sys.stdout,
    "_tee_original",
    sys.stdout,
)

ORIGINAL_STDERR = getattr(
    sys.stderr,
    "_tee_original",
    sys.stderr,
)

sys.stdout = TeeStream(
    ORIGINAL_STDOUT,
    LOG_FILE,
)

sys.stderr = TeeStream(
    ORIGINAL_STDERR,
    LOG_FILE,
)


def close_execution_log():
    """Flush and close the SI log safely at interpreter shutdown."""

    try:
        LOG_FILE.flush()
        LOG_FILE.close()
    except Exception:
        pass


atexit.register(close_execution_log)


def log_unhandled_exception(
    exception_type,
    exception_value,
    exception_traceback,
):
    """Record an unhandled exception in the same SI execution log."""

    print("\n")
    print("=" * 72)
    print("RUN TERMINATED WITH AN UNHANDLED ERROR")
    print("=" * 72)

    traceback.print_exception(
        exception_type,
        exception_value,
        exception_traceback,
        file=sys.stderr,
    )

    print(f"Execution log retained at: {LOG_PATH}")

    ORIGINAL_EXCEPTHOOK(
        exception_type,
        exception_value,
        exception_traceback,
    )


ORIGINAL_EXCEPTHOOK = sys.excepthook
sys.excepthook = log_unhandled_exception


def package_version(distribution_name):
    """Return an installed package version without failing the workflow."""

    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return "not available"
    except Exception as error:
        return f"unavailable ({error})"


def sha256_file(path, chunk_size=1024 * 1024):
    """Calculate a SHA-256 checksum for reproducibility."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as input_file:
        while True:
            chunk = input_file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def log_input_file(label, path):
    """Write input-file metadata and checksum to the execution log."""

    path = Path(path)

    print(f"{label}: {path}")
    print(f"  Exists: {path.exists()}")

    if path.exists():
        print(f"  Size_bytes: {path.stat().st_size}")
        print(f"  SHA256: {sha256_file(path)}")


print("=" * 72)
print("STRUCTURAL ERROR ANALYSIS EXECUTION LOG")
print("=" * 72)
print(f"Run start UTC: {RUN_START_UTC.isoformat()}")
print(f"Python version: {sys.version.replace(chr(10), ' ')}")
print(f"Platform: {platform.platform()}")
print(f"Executable: {sys.executable}")

print("\nSoftware versions")
print("-----------------")
print(f"NumPy: {package_version('numpy')}")
print(f"pandas: {package_version('pandas')}")
print(f"SciPy: {package_version('scipy')}")
print(f"Matplotlib: {package_version('matplotlib')}")
print(f"RDKit: {package_version('rdkit')}")

print("\nAnalysis settings")
print("-----------------")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"SELECTED_THRESHOLD: {SELECTED_THRESHOLD}")
print(f"TOP_STRUCTURES_TO_DRAW: {TOP_STRUCTURES_TO_DRAW}")
print(f"MORGAN_RADIUS: {MORGAN_RADIUS}")
print(f"MORGAN_BITS: {MORGAN_BITS}")

print("\nInput files")
print("-----------")
log_input_file("Predictions", PREDICTION_PATH)
log_input_file("Cleaned data", CLEANED_DATA_PATH)
log_input_file("Split assignments", SPLIT_PATH)
log_input_file("Selected features", SELECTED_FEATURES_PATH)

print("\nExecution output")
print("----------------")


# ============================================================
# 2. Configure Arial font
# ============================================================

try:
    fm.findfont("Arial", fallback_to_default=False)
    figure_font = "Arial"
    print("Arial font found.")

except ValueError:
    figure_font = "Liberation Sans"
    print(
        "Arial was not found. Liberation Sans is being used "
        "as a metrically similar fallback."
    )

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [figure_font],
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# 3. Check that required files exist
# ============================================================

required_files = [
    PREDICTION_PATH,
    CLEANED_DATA_PATH,
    SPLIT_PATH,
    SELECTED_FEATURES_PATH,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "The following required files were not found:\n"
        + "\n".join(missing_files)
        + "\n\nRun the final-test workflow before this analysis."
    )


# ============================================================
# 4. Load workflow outputs
# ============================================================

predictions = pd.read_csv(PREDICTION_PATH)
cleaned_data = pd.read_csv(CLEANED_DATA_PATH)
split_table = pd.read_csv(SPLIT_PATH)
selected_features_table = pd.read_csv(
    SELECTED_FEATURES_PATH
)

selected_features = (
    selected_features_table["Feature"]
    .dropna()
    .astype(str)
    .tolist()
)

required_prediction_columns = {
    "Compound_ID",
    "Canonical_SMILES",
    "y_true",
    "y_prediction",
    "y_probability_toxic",
}

missing_prediction_columns = (
    required_prediction_columns
    - set(predictions.columns)
)

if missing_prediction_columns:
    raise ValueError(
        "Missing prediction columns: "
        f"{sorted(missing_prediction_columns)}"
    )

predictions["y_true"] = pd.to_numeric(
    predictions["y_true"],
    errors="raise",
).astype(int)

predictions["y_prediction"] = pd.to_numeric(
    predictions["y_prediction"],
    errors="raise",
).astype(int)

predictions["y_probability_toxic"] = pd.to_numeric(
    predictions["y_probability_toxic"],
    errors="coerce",
)


# ============================================================
# 5. Merge predictions with molecular descriptors
# ============================================================

descriptor_columns = [
    feature
    for feature in selected_features
    if feature in cleaned_data.columns
]

core_descriptors = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "HeavyAtomCount",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeteroatoms",
    "HallKierAlpha",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi4n",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "FractionCSP3",
]

core_descriptors = [
    descriptor
    for descriptor in core_descriptors
    if descriptor in cleaned_data.columns
]

analysis_descriptors = list(
    dict.fromkeys(
        core_descriptors + descriptor_columns
    )
)

cleaned_subset_columns = list(
    dict.fromkeys(
        [
            "Compound_ID",
            "Canonical_SMILES",
            "Scaffold",
        ]
        + analysis_descriptors
    )
)

cleaned_subset_columns = [
    column
    for column in cleaned_subset_columns
    if column in cleaned_data.columns
]

analysis_data = predictions.merge(
    cleaned_data[cleaned_subset_columns],
    on="Compound_ID",
    how="left",
    suffixes=("", "_descriptor"),
)

if "Canonical_SMILES_descriptor" in analysis_data.columns:
    analysis_data["Canonical_SMILES"] = (
        analysis_data["Canonical_SMILES"]
        .fillna(
            analysis_data[
                "Canonical_SMILES_descriptor"
            ]
        )
    )

if (
    "Scaffold" not in analysis_data.columns
    and "Scaffold_descriptor" in analysis_data.columns
):
    analysis_data["Scaffold"] = (
        analysis_data["Scaffold_descriptor"]
    )


# ============================================================
# 6. Identify false negatives and true positives
# ============================================================

analysis_data["Error_group"] = "Other"

analysis_data.loc[
    (
        (analysis_data["y_true"] == 1)
        & (analysis_data["y_prediction"] == 0)
    ),
    "Error_group",
] = "False negative"

analysis_data.loc[
    (
        (analysis_data["y_true"] == 1)
        & (analysis_data["y_prediction"] == 1)
    ),
    "Error_group",
] = "True positive"

toxic_data = analysis_data[
    analysis_data["y_true"] == 1
].copy()

false_negatives = toxic_data[
    toxic_data["Error_group"] == "False negative"
].copy()

true_positives = toxic_data[
    toxic_data["Error_group"] == "True positive"
].copy()

print("\nToxic-compound classification")
print("-----------------------------")
print(f"Total toxic compounds: {len(toxic_data)}")
print(f"False negatives:       {len(false_negatives)}")
print(f"True positives:        {len(true_positives)}")
print(
    "Calculated toxic recall: "
    f"{len(true_positives) / len(toxic_data):.3f}"
)

if len(false_negatives) == 0:
    raise ValueError(
        "No false-negative toxic compounds were identified."
    )

if len(true_positives) == 0:
    raise ValueError(
        "No true-positive toxic compounds were identified."
    )


# ============================================================
# 7. Calculate structural similarity to development compounds
# ============================================================

development_ids = split_table.loc[
    split_table["Partition"] == "development",
    "Compound_ID",
]

development_data = cleaned_data[
    cleaned_data["Compound_ID"].isin(
        development_ids
    )
].copy()

if "Label" not in development_data.columns:
    raise ValueError(
        "The cleaned dataset must contain the Label column."
    )

fingerprint_generator = (
    rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
    )
)


def molecule_and_fingerprint(smiles):
    """Return an RDKit molecule and Morgan fingerprint."""

    molecule = Chem.MolFromSmiles(str(smiles))

    if molecule is None:
        return None, None

    fingerprint = (
        fingerprint_generator.GetFingerprint(
            molecule
        )
    )

    return molecule, fingerprint


development_molecules = []
development_fingerprints = []
development_labels = []
development_compound_ids = []

for _, row in development_data.iterrows():
    molecule, fingerprint = (
        molecule_and_fingerprint(
            row["Canonical_SMILES"]
        )
    )

    if fingerprint is None:
        continue

    development_molecules.append(molecule)
    development_fingerprints.append(fingerprint)
    development_labels.append(int(row["Label"]))
    development_compound_ids.append(
        row["Compound_ID"]
    )

development_labels = np.asarray(
    development_labels,
    dtype=int,
)

toxic_development_indices = np.where(
    development_labels == 1
)[0]

nontoxic_development_indices = np.where(
    development_labels == 0
)[0]


def nearest_development_information(smiles):
    """Calculate similarity to development compounds."""

    molecule, fingerprint = (
        molecule_and_fingerprint(smiles)
    )

    if fingerprint is None:
        return pd.Series(
            {
                "max_similarity_all_development": np.nan,
                "nearest_development_compound": None,
                "nearest_development_label": np.nan,
                "max_similarity_toxic_development": np.nan,
                "max_similarity_nontoxic_development": np.nan,
                "toxic_similarity_margin": np.nan,
            }
        )

    similarities = np.asarray(
        DataStructs.BulkTanimotoSimilarity(
            fingerprint,
            development_fingerprints,
        ),
        dtype=float,
    )

    nearest_index = int(
        np.argmax(similarities)
    )

    max_all = float(
        similarities[nearest_index]
    )

    max_toxic = float(
        similarities[
            toxic_development_indices
        ].max()
    )

    max_nontoxic = float(
        similarities[
            nontoxic_development_indices
        ].max()
    )

    return pd.Series(
        {
            "max_similarity_all_development": max_all,
            "nearest_development_compound": (
                development_compound_ids[
                    nearest_index
                ]
            ),
            "nearest_development_label": int(
                development_labels[
                    nearest_index
                ]
            ),
            "max_similarity_toxic_development": (
                max_toxic
            ),
            "max_similarity_nontoxic_development": (
                max_nontoxic
            ),
            "toxic_similarity_margin": (
                max_toxic - max_nontoxic
            ),
        }
    )


print("\nCalculating similarity to development compounds...")

similarity_results = toxic_data[
    "Canonical_SMILES"
].apply(
    nearest_development_information
)

toxic_data = pd.concat(
    [
        toxic_data.reset_index(drop=True),
        similarity_results.reset_index(drop=True),
    ],
    axis=1,
)

false_negatives = toxic_data[
    toxic_data["Error_group"] == "False negative"
].copy()

true_positives = toxic_data[
    toxic_data["Error_group"] == "True positive"
].copy()


# ============================================================
# 8. Screen interpretable functional groups
# ============================================================

functional_group_smarts = {
    "Halogen": "[F,Cl,Br,I]",
    "Nitro": "[N+](=O)[O-]",
    "Carboxylic_acid": "C(=O)[O;H,-]",
    "Ester": "C(=O)O[#6]",
    "Amide": "C(=O)N",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Nitrile": "C#N",
    "Primary_or_secondary_amine": (
        "[NX3;H2,H1;!$(NC=O)]"
    ),
    "Tertiary_amine": (
        "[NX3;H0;!$(N-*=[O,N,P,S])]"
    ),
    "Quaternary_ammonium": "[N+;X4]",
    "Phenol": "[OX2H]-c",
    "Thiol": "[SX2H]",
    "Thioether": "[#6]-[SX2]-[#6]",
    "Sulfonamide": "S(=O)(=O)N",
    "Isocyanate": "N=C=O",
    "Acyl_halide": "C(=O)[F,Cl,Br,I]",
    "Epoxide": "[OX2r3]1[CX4r3][CX4r3]1",
    "Phosphate": "P(=O)(O)(O)",
    "Aromatic_atom": "a",
}

compiled_patterns = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts
    in functional_group_smarts.items()
}


def functional_group_presence(smiles, pattern):
    molecule = Chem.MolFromSmiles(str(smiles))

    if molecule is None or pattern is None:
        return np.nan

    return int(
        molecule.HasSubstructMatch(pattern)
    )


for group_name, pattern in compiled_patterns.items():
    column_name = f"FG_{group_name}"

    toxic_data[column_name] = toxic_data[
        "Canonical_SMILES"
    ].apply(
        lambda smiles: functional_group_presence(
            smiles,
            pattern,
        )
    )

false_negatives = toxic_data[
    toxic_data["Error_group"] == "False negative"
].copy()

true_positives = toxic_data[
    toxic_data["Error_group"] == "True positive"
].copy()


# ============================================================
# 9. Statistical helper functions
# ============================================================

def cliffs_delta(group_a, group_b):
    """
    Cliff's delta.

    Positive values mean the descriptor is generally higher
    among false negatives.
    """

    group_a = np.asarray(
        group_a,
        dtype=float,
    )

    group_b = np.asarray(
        group_b,
        dtype=float,
    )

    differences = (
        group_a[:, None]
        - group_b[None, :]
    )

    greater = np.sum(differences > 0)
    lower = np.sum(differences < 0)

    return (
        (greater - lower)
        / differences.size
    )


def benjamini_hochberg(p_values):
    """Benjamini–Hochberg false-discovery correction."""

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
    )

    valid_mask = np.isfinite(p_values)

    if valid_mask.sum() == 0:
        return adjusted

    valid_p = p_values[valid_mask]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]

    number_of_tests = len(ranked_p)
    ranks = np.arange(
        1,
        number_of_tests + 1,
    )

    ranked_adjusted = (
        ranked_p
        * number_of_tests
        / ranks
    )

    ranked_adjusted = np.minimum.accumulate(
        ranked_adjusted[::-1]
    )[::-1]

    ranked_adjusted = np.clip(
        ranked_adjusted,
        0,
        1,
    )

    valid_adjusted = np.empty(
        number_of_tests
    )

    valid_adjusted[order] = ranked_adjusted
    adjusted[valid_mask] = valid_adjusted

    return adjusted


# ============================================================
# 10. Compare descriptor distributions
# ============================================================

descriptor_rows = []

for descriptor in analysis_descriptors:
    if descriptor not in toxic_data.columns:
        continue

    false_negative_values = pd.to_numeric(
        false_negatives[descriptor],
        errors="coerce",
    ).dropna()

    true_positive_values = pd.to_numeric(
        true_positives[descriptor],
        errors="coerce",
    ).dropna()

    if (
        len(false_negative_values) < 3
        or len(true_positive_values) < 3
    ):
        continue

    try:
        _, p_value = mannwhitneyu(
            false_negative_values,
            true_positive_values,
            alternative="two-sided",
        )
    except ValueError:
        p_value = np.nan

    delta = cliffs_delta(
        false_negative_values.values,
        true_positive_values.values,
    )

    descriptor_rows.append(
        {
            "Descriptor": descriptor,
            "False_negative_N": len(
                false_negative_values
            ),
            "False_negative_median": (
                false_negative_values.median()
            ),
            "False_negative_Q1": (
                false_negative_values.quantile(0.25)
            ),
            "False_negative_Q3": (
                false_negative_values.quantile(0.75)
            ),
            "True_positive_N": len(
                true_positive_values
            ),
            "True_positive_median": (
                true_positive_values.median()
            ),
            "True_positive_Q1": (
                true_positive_values.quantile(0.25)
            ),
            "True_positive_Q3": (
                true_positive_values.quantile(0.75)
            ),
            "Median_difference_FN_minus_TP": (
                false_negative_values.median()
                - true_positive_values.median()
            ),
            "Cliffs_delta": delta,
            "Absolute_Cliffs_delta": abs(delta),
            "Mann_Whitney_p": p_value,
        }
    )

descriptor_comparison = pd.DataFrame(
    descriptor_rows
)

if not descriptor_comparison.empty:
    descriptor_comparison[
        "BH_adjusted_q"
    ] = benjamini_hochberg(
        descriptor_comparison[
            "Mann_Whitney_p"
        ].values
    )

    descriptor_comparison = (
        descriptor_comparison.sort_values(
            by="Absolute_Cliffs_delta",
            ascending=False,
        )
        .reset_index(drop=True)
    )

descriptor_output_path = (
    OUTPUT_DIR
    / "false_negative_descriptor_comparison.csv"
)

descriptor_comparison.to_csv(
    descriptor_output_path,
    index=False,
)


# ============================================================
# 11. Compare functional-group frequencies
# ============================================================

functional_group_rows = []

functional_group_columns = [
    column
    for column in toxic_data.columns
    if column.startswith("FG_")
]

for column in functional_group_columns:
    fn_present = int(
        false_negatives[column].fillna(0).sum()
    )

    tp_present = int(
        true_positives[column].fillna(0).sum()
    )

    fn_absent = (
        len(false_negatives) - fn_present
    )

    tp_absent = (
        len(true_positives) - tp_present
    )

    contingency_table = [
        [fn_present, fn_absent],
        [tp_present, tp_absent],
    ]

    odds_ratio, p_value = fisher_exact(
        contingency_table,
        alternative="two-sided",
    )

    functional_group_rows.append(
        {
            "Functional_group": column.replace(
                "FG_",
                "",
            ),
            "False_negative_present_N": fn_present,
            "False_negative_prevalence": (
                fn_present / len(false_negatives)
            ),
            "True_positive_present_N": tp_present,
            "True_positive_prevalence": (
                tp_present / len(true_positives)
            ),
            "Prevalence_difference_FN_minus_TP": (
                fn_present / len(false_negatives)
                - tp_present / len(true_positives)
            ),
            "Odds_ratio": odds_ratio,
            "Fisher_exact_p": p_value,
        }
    )

functional_group_comparison = pd.DataFrame(
    functional_group_rows
)

functional_group_comparison[
    "BH_adjusted_q"
] = benjamini_hochberg(
    functional_group_comparison[
        "Fisher_exact_p"
    ].values
)

functional_group_comparison = (
    functional_group_comparison.sort_values(
        by="Prevalence_difference_FN_minus_TP",
        key=lambda values: values.abs(),
        ascending=False,
    )
    .reset_index(drop=True)
)

functional_group_output_path = (
    OUTPUT_DIR
    / "false_negative_functional_group_comparison.csv"
)

functional_group_comparison.to_csv(
    functional_group_output_path,
    index=False,
)


# ============================================================
# 12. Generate group-level summary
# ============================================================

def group_summary(group_name, group_data):
    row = {
        "Group": group_name,
        "N": len(group_data),
        "Median_predicted_toxic_probability": (
            group_data[
                "y_probability_toxic"
            ].median()
        ),
        "Probability_Q1": (
            group_data[
                "y_probability_toxic"
            ].quantile(0.25)
        ),
        "Probability_Q3": (
            group_data[
                "y_probability_toxic"
            ].quantile(0.75)
        ),
        "Median_max_similarity_all_development": (
            group_data[
                "max_similarity_all_development"
            ].median()
        ),
        "Median_similarity_to_toxic_development": (
            group_data[
                "max_similarity_toxic_development"
            ].median()
        ),
        "Median_similarity_to_nontoxic_development": (
            group_data[
                "max_similarity_nontoxic_development"
            ].median()
        ),
        "Median_toxic_similarity_margin": (
            group_data[
                "toxic_similarity_margin"
            ].median()
        ),
    }

    if (
        "inside_applicability_domain"
        in group_data.columns
    ):
        inside_values = (
            group_data[
                "inside_applicability_domain"
            ]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

        row["Outside_AD_N"] = int(
            (inside_values == False).sum()
        )

        row["Outside_AD_percent"] = (
            100
            * (inside_values == False).mean()
        )

    if "mean_kNN_distance" in group_data.columns:
        row["Median_kNN_distance"] = (
            pd.to_numeric(
                group_data["mean_kNN_distance"],
                errors="coerce",
            ).median()
        )

    return row


summary_table = pd.DataFrame(
    [
        group_summary(
            "False negative",
            false_negatives,
        ),
        group_summary(
            "True positive",
            true_positives,
        ),
    ]
)

summary_output_path = (
    OUTPUT_DIR
    / "false_negative_group_summary.csv"
)

summary_table.to_csv(
    summary_output_path,
    index=False,
)


# ============================================================
# 13. Classify false negatives by confidence
# ============================================================

false_negatives[
    "False_negative_type"
] = np.select(
    [
        false_negatives[
            "y_probability_toxic"
        ] >= SELECTED_THRESHOLD - 0.10,

        false_negatives[
            "y_probability_toxic"
        ] < 0.30,
    ],
    [
        "Borderline false negative",
        "Confident false negative",
    ],
    default="Intermediate false negative",
)

false_negative_output_columns = [
    "Compound_ID",
    "Canonical_SMILES",
    "Scaffold",
    "y_true",
    "y_prediction",
    "y_probability_toxic",
    "False_negative_type",
    "inside_applicability_domain",
    "mean_kNN_distance",
    "max_similarity_all_development",
    "nearest_development_compound",
    "nearest_development_label",
    "max_similarity_toxic_development",
    "max_similarity_nontoxic_development",
    "toxic_similarity_margin",
]

false_negative_output_columns += (
    core_descriptors
)

false_negative_output_columns += (
    functional_group_columns
)

false_negative_output_columns = list(
    dict.fromkeys(
        column
        for column
        in false_negative_output_columns
        if column in false_negatives.columns
    )
)

false_negative_compound_output_path = (
    OUTPUT_DIR
    / "false_negative_compound_level_analysis.csv"
)

false_negatives[
    false_negative_output_columns
].sort_values(
    by="y_probability_toxic",
    ascending=True,
).to_csv(
    false_negative_compound_output_path,
    index=False,
)


# ============================================================
# 14. Create descriptor effect-size figure
# ============================================================

if not descriptor_comparison.empty:
    top_effects = (
        descriptor_comparison.head(12)
        .sort_values(
            "Cliffs_delta",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 6.5)
    )

    ax.barh(
        top_effects["Descriptor"],
        top_effects["Cliffs_delta"],
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Cliff's delta: false negatives versus true positives",
        fontfamily=figure_font,
    )

    ax.set_ylabel(
        "Molecular descriptor",
        fontfamily=figure_font,
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
    )

    for label in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        label.set_fontfamily(
            figure_font
        )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    effect_png_path = (
        OUTPUT_DIR
        / "false_negative_descriptor_effects.png"
    )

    effect_pdf_path = (
        OUTPUT_DIR
        / "false_negative_descriptor_effects.pdf"
    )

    fig.savefig(
        effect_png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        effect_pdf_path,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 15. Create a grid of the most confident false negatives
# ============================================================

structure_subset = (
    false_negatives.sort_values(
        by="y_probability_toxic",
        ascending=True,
    )
    .head(TOP_STRUCTURES_TO_DRAW)
)

structure_molecules = []
structure_legends = []

for _, row in structure_subset.iterrows():
    molecule = Chem.MolFromSmiles(
        str(row["Canonical_SMILES"])
    )

    if molecule is None:
        continue

    structure_molecules.append(molecule)

    ad_text = "AD unavailable"

    if (
        "inside_applicability_domain" in row.index
        and pd.notna(row["inside_applicability_domain"])
    ):
        inside_text = str(
            row["inside_applicability_domain"]
        ).strip().lower()

        ad_text = (
            "Inside AD"
            if inside_text in ["true", "1", "yes"]
            else "Outside AD"
        )

    # Maximum similarity to development compounds
    similarity = row.get(
        "max_similarity_all_development",
        np.nan,
    )

    similarity_text = (
        f"{float(similarity):.2f}"
        if pd.notna(similarity)
        else "NA"
    )

    compound_id = row.get(
        "Compound_ID",
        "Unknown compound",
    )

    probability = float(
        row["y_probability_toxic"]
    )

    structure_legends.append(
        f"{compound_id}\n"
        f"P(toxic) = {probability:.2f}\n"
        f"{ad_text}; similarity = {similarity_text}"
    )


# Create and save the structure grid
structure_grid_path = (
    OUTPUT_DIR
    / "false_negative_structure_grid.png"
)

if structure_molecules:

    # returnPNG=True returns PNG binary data rather than
    # an IPython display object
    structure_grid_png = Draw.MolsToGridImage(
        structure_molecules,
        molsPerRow=4,
        subImgSize=(320, 260),
        legends=structure_legends,
        useSVG=False,
        returnPNG=True,
    )

    # Some RDKit notebook versions still wrap the PNG bytes
    # inside an IPython Image object
    if hasattr(structure_grid_png, "data"):
        structure_grid_png = structure_grid_png.data

    if isinstance(structure_grid_png, memoryview):
        structure_grid_png = (
            structure_grid_png.tobytes()
        )

    if not isinstance(
        structure_grid_png,
        (bytes, bytearray),
    ):
        raise TypeError(
            "RDKit did not return PNG binary data. "
            f"Returned type: {type(structure_grid_png)}"
        )

    structure_grid_path.write_bytes(
        structure_grid_png
    )

    print(
        "Structure grid created successfully:"
    )
    print(structure_grid_path)

else:
    print(
        "No valid molecular structures were available "
        "for the false-negative grid."
    )


# ============================================================
# 16. Print the principal results
# ============================================================

print("\nGroup-level summary")
print("-------------------")
print(summary_table.to_string(index=False))

print("\nDescriptors with the largest effect sizes")
print("------------------------------------------")

if not descriptor_comparison.empty:
    print(
        descriptor_comparison[
            [
                "Descriptor",
                "False_negative_median",
                "True_positive_median",
                "Cliffs_delta",
                "Mann_Whitney_p",
                "BH_adjusted_q",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )

print("\nLargest functional-group differences")
print("------------------------------------")
print(
    functional_group_comparison[
        [
            "Functional_group",
            "False_negative_prevalence",
            "True_positive_prevalence",
            "Prevalence_difference_FN_minus_TP",
            "Odds_ratio",
            "BH_adjusted_q",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nSaved files")
print("-----------")

saved_paths = [
    descriptor_output_path,
    functional_group_output_path,
    summary_output_path,
    false_negative_compound_output_path,
    OUTPUT_DIR
    / "false_negative_descriptor_effects.png",
    OUTPUT_DIR
    / "false_negative_descriptor_effects.pdf",
    OUTPUT_DIR
    / "false_negative_structure_grid.png",
    LOG_PATH,
]

for path in saved_paths:
    print(
        f"{path.name}: "
        f"{'created' if path.exists() else 'not created'}"
    )

# ============================================================
# 17. Finalize the Supporting Information execution log
# ============================================================

RUN_END_UTC = datetime.now(timezone.utc)
RUN_DURATION_SECONDS = (
    RUN_END_UTC - RUN_START_UTC
).total_seconds()

print("\n")
print("=" * 72)
print("RUN COMPLETED SUCCESSFULLY")
print("=" * 72)
print(f"Run end UTC: {RUN_END_UTC.isoformat()}")
print(f"Run duration seconds: {RUN_DURATION_SECONDS:.2f}")
print(f"Execution log saved to: {LOG_PATH}")

LOG_FILE.flush()

