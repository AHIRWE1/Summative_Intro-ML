# Predicting Hospital Readmissions for Diabetic Patients


## Project Overview
Hospital readmissions within 30 days of discharge represent a critical challenge for healthcare quality and cost management. This project implements an end-to-end machine learning lifecycle from rigorous patient-grouped data splitting and feature preprocessing to systematic experimentation comparing traditional machine learning models and deep neural networks.

## Key Features & Methodology
* **Patient-Grouped Splitting:** Implemented strict patient-level grouping (`GroupShuffleSplit`) to prevent data leakage across train, validation, and test splits.
* **Feature Engineering & Preprocessing:** Handled missing values using training-fitted imputation pipelines, applied one-hot encoding with minimum frequency thresholds, and standardized numerical attributes.
* **Systematic Experimentation (7 Models):** Evaluated traditional models (Logistic Regression, Random Forest, hyperparameter-tuned Random Forest via `GroupKFold`) alongside deep learning architectures (Baseline NN, varied learning rates, dropout regularization, and compact neural network variants).
* **Evaluation & Error Analysis:** Assessed performance using ranking metrics (Average Precision / PR-AUC and ROC-AUC) alongside confusion matrix breakdowns and demographic subgroup reviews.
* **Experiment Tracking:** Integrated with Weights & Biases (W&B) for offline/online tracking, metric logging, and artifact management.

## Repository Structure
```
├── diabetes_130us_readmission_analysis.ipynb   # Main Jupyter/Colab notebook containing all code & narrative
├── requirements.txt                         # Python package dependencies
└── README.md                                # Project documentation
```

## Data and Artifacts Notice
* **Raw Dataset:** Due to storage constraints and privacy considerations, the raw dataset (`dataset_diabetes/`) is not stored directly in this repository. 
* **Model Artifacts & Reports:** Saved models (`.keras`, `.weights.h5`, `.joblib`) and evaluation reports (`.csv`, `.json`) are hosted externally in a secure Google Drive directory.

## Reproducibility
1. Clone this repository:
   ```bash
   git clone https://github.com/AHIRWE1/Summative_Intro-ML.git 
   cd Summative_Intro-ML
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `hospital_readmission_ml_pipeline.ipynb` in Google Colab or a local Jupyter environment.
