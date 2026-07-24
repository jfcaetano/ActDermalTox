# ActDermalTox 🧪
Repository for the article "Machine Learning for Go/No-Go Screening of Acute Dermal Toxicity for Sustainable Chemical Design"

<div align="center">
  <img src="Supporting Information/Graphical.png" 
       alt="Screenshot" width="433" height="272">
</div>


## Original Source Dataset
The original dataset used in this model is derived from:
Koutroumpa, N.M. (2025). Dataset ds13 - Acute dermal toxicity of small molecules. NovaMechanics. DOI : https://db.chempharos.eu/datasets/Datasets.zul?datasetID=ds13

## Scripts
Execute in Google Collab in the folliwng order:

# Main Model Workflow

An end-to-end Google Colab workflow for building and validating binary toxicity classification models from molecular structures.

The pipeline predicts:

* **GO (0):** predicted non-toxic
* **NO-GO (1):** predicted toxic

## What the Workflow Does

* Cleans, canonicalizes, and deduplicates molecular structures.
* Calculates physicochemical descriptors using RDKit.
* Creates a **scaffold-separated locked test set** to reduce structural data leakage.
* Compares Logistic Regression, SVM, Random Forest, Gradient Boosting, and XGBoost using identical scaffold-grouped cross-validation folds.
* Tunes XGBoost using development data only.
* Selects and freezes the 60 most important molecular descriptors.
* Calculates Variance Inflation Factors to assess descriptor multicollinearity.
* Selects a fixed classification threshold from out-of-fold predictions.
* Compares three class-imbalance strategies:

  * `scale_pos_weight`
  * SMOTE
  * ADASYN
* Evaluates the frozen models once on the locked scaffold test set.
* Calculates an applicability domain using nearest-neighbour distances.
* Generates SHAP explanations for global and compound-level model interpretation.
* Saves models, predictions, metrics, plots, configurations, software versions, and execution logs.

## Two-Stage Execution

### `development`

Performs model comparison, tuning, feature selection, VIF analysis, threshold selection, and development-only imbalance comparisons. The locked test set remains untouched.

### `final_test`

Loads the frozen development configuration, trains the final weighted, SMOTE, and ADASYN models, and evaluates the locked test set once. It also produces applicability-domain and SHAP analyses.

> Run `development` first, archive the frozen outputs, then change only `RUN_MODE` to `final_test`. Do not retune the model after inspecting the locked-test results.

## Additional Analysis Scripts

### Threshold Performance Analysis

`thereshold-performance.py` examines the development threshold search and visualizes how toxic-class precision, recall, and F1-score change across classification thresholds. It identifies the optimal operating point subject to the minimum precision requirement and exports publication-ready PNG and PDF figures.

### Applicability-Domain Performance

`applicability-domain.py` evaluates model performance separately for compounds inside and outside the applicability domain. It reports confusion-matrix counts, classification metrics, ROC-AUC, PR-AUC, MCC, and Brier score for the complete locked test set and both applicability-domain subsets.

### Structural Error Analysis

`structural-analysis.py` investigates toxic compounds incorrectly classified as non-toxic. It:

* Compares false negatives with correctly identified toxic compounds.
* Calculates Morgan-fingerprint similarity to development compounds.
* Screens interpretable functional groups.
* Tests molecular descriptor and functional-group differences.
* Applies effect-size analysis and multiple-testing correction.
* Classifies false negatives by prediction confidence.
* Generates descriptor plots and molecular-structure grids.
* Records software versions, file checksums, settings, and execution logs for reproducibility.

This analysis helps identify structural patterns, chemical functionalities, and applicability-domain limitations associated with model failures.

## Analysis Flow

```text
Development workflow
        ↓
Threshold selection and visualization
        ↓
Locked scaffold-test evaluation
        ↓
Applicability-domain performance analysis
        ↓
Structural investigation of false negatives
```

