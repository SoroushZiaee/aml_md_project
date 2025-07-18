# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project focused on analyzing AML (Acute Myeloid Leukemia) and MDS (Myelodysplastic Syndrome) patient data to predict transplant outcomes. The project investigates how dynamic changes in CD3+ chimerism levels at different time points (Day 30, 60, and 100) can predict allogeneic stem cell transplant outcomes, particularly:

- Disease relapse (primary focus)
- Acute graft-versus-host disease (aGVHD)
- Chronic graft-versus-host disease (cGVHD)
- Overall survival (OS)
- Relapse-free survival (RFS)
- GVHD-free, relapse-free survival (GRFS)

## Development Environment

### Python Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning models and preprocessing
- matplotlib, seaborn, plotly: Data visualization
- jupyter: Notebook environment
- imbalanced-learn: Handling class imbalance with SMOTE
- lifelines: Survival analysis
- joblib: Model serialization

### Running Jupyter Notebooks
Start Jupyter environment:
```bash
jupyter notebook
# or
jupyter lab
```

## Key Datasets and Files

### Input Data
- `main_dataset.xlsx`: Primary dataset with patient information
- `preprocessed_ml_for_aml_mds.csv/.xlsx`: Cleaned and preprocessed data
- `CD_file.csv`: Subset focusing on chimerism dynamics and outcomes

### Analysis Notebooks
- `aml-mds.ipynb`: Main analysis notebook with comprehensive ML pipeline
- `basic_analysis_on_aml_mds.ipynb`: Exploratory data analysis and feature engineering
- `ROC_curve.ipynb`: ROC curve analysis for model evaluation
- `regression_model.ipynb`: Regression modeling for continuous outcomes
- `visulization_chimerism_*.ipynb`: Chimerism trend visualizations

## Core Architecture

### Feature Engineering Pipeline
The project implements sophisticated feature engineering for chimerism dynamics:

1. **Dynamic Change Features**: Calculate differences between time points
   - `d(30-60)_cd3+/cd3-`: Change from Day 30 to Day 60
   - `d(60-100)_cd3+/cd3-`: Change from Day 60 to Day 100

2. **Statistical Features**: Aggregate measures across time points
   - `mean_ch+/ch-`: Mean chimerism levels
   - `std_ch+/ch-`: Standard deviation (variability)
   - `cv_ch+/ch-`: Coefficient of variation (normalized variability)

3. **Categorical Encoding**: Convert dynamic patterns to labels
   - "upward", "downward", "fluctuate", "no_change", "unknown"

### Machine Learning Pipeline

#### Feature Selection Methods
- `SelectKBest`: Statistical feature selection using f_classif/f_regression
- `RFE`: Recursive Feature Elimination
- `feature_importance`: Tree-based importance filtering
- Custom optimization with Fibonacci sequence k values: [1, 2, 3, 5, 8, 13, 21, 34]

#### Model Training Architecture
Key functions in notebooks:
- `train_classification_models()`: Multi-label classification with GridSearchCV
- `optimize_feature_selection()`: Automated feature selection optimization
- `plot_classification_results()`: Comprehensive results visualization
- `select_features_and_train()`: Train models with specific feature sets

#### Data Preprocessing
- Multiple imputation strategies (SimpleImputer, KNNImputer, IterativeImputer)
- StandardScaler for feature normalization
- LabelEncoder for categorical variables
- Handling missing values with domain-specific logic

### Model Evaluation Framework
- Cross-validation with KFold (typically 5-fold)
- Multiple metrics: accuracy, F1-score, precision, recall
- Confusion matrices and classification reports
- ROC curves and AUC analysis
- Feature importance visualization

## Saved Models

Trained models are stored in the `models/` directory with systematic naming:
- `Random_Forest_k{n}.joblib`: Models with different feature counts
- `y_{outcome}_{algorithm}_{features}_{smote}.joblib`: Outcome-specific models

Where:
- `{outcome}`: agvhd, cgvhd, death, relapse, rfs
- `{algorithm}`: Random_Forest
- `{features}`: Feature selection variant (k1, k2, etc.)
- `{smote}`: With/without SMOTE oversampling

## Key Research Questions

### Tier 1 (Primary)
Can dynamic CD3+ chimerism changes predict disease relapse? Focus on trend patterns:
- Upward trend (Day 30→100)
- Increase then decrease (Day 30→60→100)
- Percentage changes (≥20% increase)

### Tier 2
Prediction of other outcomes: OS, GVHD, GRFS using chimerism dynamics

### Tier 3
Interaction effects between CD3+ chimerism and other biomarkers:
- CD3- chimerism dynamics
- MRD (Measurable Residual Disease) status
- GVHD presence

## Development Notes

### Data Filtering
- Filter by `disease == 1` for AML/MDS patients
- Remove columns with excessive missing data
- Handle DLI (Donor Lymphocyte Infusion) as intervention vs prevention

### Feature Importance Insights
Most predictive features consistently include:
- `d(60-100)_cd3+`: Change in CD3+ chimerism from Day 60 to 100
- `d100_cd3-`: CD3- chimerism at Day 100
- `std_ch+/std_ch-`: Chimerism variability measures

### Model Performance
- Best single feature: `d(60-100)_cd3+` (accuracy ~74%)
- Optimal feature count: k=1-5 features for most outcomes
- Random Forest consistently outperforms SVM and Naive Bayes

### Visualization Outputs
Results saved in `plots/` directories:
- Feature importance plots
- ROC curves  
- Precision-recall curves
- Performance comparison charts