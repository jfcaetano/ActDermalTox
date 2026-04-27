# ActDermalTox
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

### 1 Model_Algorithm_Screening.py
Google Colab initial workflow comparing Random Forest, Gradient Boosting, and XGBoost toxicity models using RDKit descriptors, hyperparameter tuning, threshold optimization, and final test-set ranking.

### 2 Model_XGB_Screening.py
Google Colab XGBoost workflow for toxicity screening, with improved tuning, validation, feature selection, applicability-domain correction, and threshold optimization.

### 3 Main_Model_Dermal.py
Google Colab XGBoost model for toxicity screening, classifying compounds as GO/non-toxic or NO-GO/toxic using RDKit descriptors, cross-validation, applicability-domain analysis, and saved performance outputs.

### 4 Main_Model_Dermal_SHAP.py
Google Colab XGBoost toxicity model with GO/NO-GO prediction, top-feature selection, and SHAP analysis to explain the most influential molecular descriptors.
