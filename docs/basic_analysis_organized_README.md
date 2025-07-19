# Basic Analysis Organized Notebook - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Requirements](#data-requirements)
4. [Notebook Structure](#notebook-structure)
5. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
6. [Advanced Machine Learning Features](#advanced-machine-learning-features)
7. [Output Files and Results](#output-files-and-results)
8. [Troubleshooting](#troubleshooting)
9. [Technical Details](#technical-details)

## Overview

The `basic_analysis_organized.ipynb` notebook is a comprehensive machine learning analysis pipeline for AML/MDS chimerism dynamics research. It combines foundational analysis with advanced ML techniques to predict transplant outcomes based on CD3+ and CD3- chimerism measurements at Day 30, 60, and 100 post-transplant.

### Research Questions Addressed
- **Primary**: Can dynamic CD3+ chimerism changes predict disease relapse?
- **Secondary**: Can chimerism dynamics predict other outcomes (OS, GVHD, GRFS)?
- **Advanced**: Which feature combinations and ML methods provide optimal prediction?

### Key Features
- ✅ **Foundational Analysis**: Feature engineering, pattern classification, EDA
- ✅ **Advanced Clustering**: K-Means and Fuzzy C-Means with UMAP 3D visualization
- ✅ **Fuzzy Classification**: SVM with probability distributions and confidence analysis
- ✅ **Genetic Algorithm**: Evolutionary feature selection with K-fold validation
- ✅ **Comprehensive Reporting**: Multi-method comparison and CSV exports

## Prerequisites

### Required Software
- Python 3.8+
- Jupyter Notebook or JupyterLab
- 8GB+ RAM recommended for advanced ML features

### Required Python Packages
Install using: `pip install -r requirements.txt`

```txt
# Core Analysis
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Advanced ML
umap-learn==0.5.3
scikit-fuzzy==0.4.2
deap==1.3.3

# Visualization
plotly>=5.0.0
ipywidgets>=8.0.0

# Data Processing
joblib>=1.3.0
lifelines>=0.27.0
imbalanced-learn>=0.11.0
```

## Data Requirements

### Primary Dataset
- **File**: `preprocessed_ml_for_aml_mds.csv`
- **Location**: Project root directory
- **Required Columns**:
  - Chimerism measurements: `d30_cd3+`, `d60_cd3+`, `d100_cd3+`, `d30_cd3-`, `d60_cd3-`, `d100_cd3-`
  - Outcomes: `y_relapse`, `y_death`, `y_agvhd`, `y_cgvhd`, `y_rfs`
  - Patient demographics: `age`, `disease`, etc.

### Data Quality Requirements
- Minimum 50 patients with complete chimerism data
- At least 10 positive cases per outcome for meaningful analysis
- Missing values <50% per column (handled by imputation)

## Notebook Structure

### Section Overview
```
1. Environment Setup (Cells 1-2)
2. Data Loading & Inspection (Cell 3)
3. Feature Engineering (Cells 4-7)
4. Exploratory Data Analysis (Cells 8-9)
5. Data Preprocessing (Cells 10-11)
6. Basic ML Analysis (Cells 12-17)
7. Advanced Clustering (Cells 18-20)
8. Fuzzy Classification (Cells 21-23)
9. Genetic Algorithm (Cells 24-25)
10. Comprehensive Summary (Cell 26)
```

## Step-by-Step Execution Guide

### Phase 1: Initial Setup and Data Loading

#### Step 1.1: Environment Setup (Cell 1-2)
```python
# Execute cells 1-2 to import all required libraries
# This includes both basic and advanced ML libraries
```

**Expected Output**: 
- ✅ Libraries imported successfully
- 📅 Analysis timestamp
- No error messages

**Troubleshooting**: 
- If import errors occur, install missing packages: `pip install <package-name>`
- For advanced ML libraries, use exact versions from requirements.txt

#### Step 1.2: Data Loading (Cell 3)
```python
# Loads preprocessed_ml_for_aml_mds.csv
df = load_and_inspect_data()
```

**Expected Output**:
- Dataset shape (e.g., 258 rows × 64 columns)
- Memory usage information
- Column type summary
- First 3 rows preview

**Common Issues**:
- **FileNotFoundError**: Ensure `preprocessed_ml_for_aml_mds.csv` is in root directory
- **Encoding errors**: File should be UTF-8 encoded

### Phase 2: Feature Engineering and Pattern Analysis

#### Step 2.1: Chimerism Feature Engineering (Cell 4-5)
```python
# Creates dynamic features from raw chimerism measurements
df_enhanced = create_chimerism_dynamics_features(df)
```

**Key Features Created**:
- **Time-point differences**: `d(30-60)_cd3+`, `d(60-100)_cd3+`
- **Statistical summaries**: `mean_cd3+`, `std_cd3+`, `cv_cd3+`
- **Slope calculations**: Linear trends over time
- **Percentage changes**: Relative changes between timepoints

**Expected Output**:
- ~25-30 new features created
- Summary of feature types
- No missing values in key features

#### Step 2.2: Pattern Classification (Cell 6-7)
```python
# Categorizes chimerism trends into clinical patterns
df_with_patterns = create_pattern_features(df_enhanced)
```

**Pattern Types**:
- `consistently_upward`: Continuous increase
- `consistently_downward`: Continuous decrease
- `fluctuating`: Up then down or vice versa
- `stable`: Minimal changes
- `mixed`: Complex patterns

**Expected Output**:
- Pattern distribution statistics
- Sample pattern assignments
- Categorical encoding for ML models

### Phase 3: Exploratory Data Analysis

#### Step 3.1: Outcome Association Analysis (Cell 8-9)
```python
# Analyzes pattern-outcome relationships
analyze_pattern_outcome_associations(df_with_patterns)
create_association_visualization(df_with_patterns)
```

**Outputs**:
- Cross-tabulation tables (pattern vs outcome)
- Proportion analyses
- Bar charts showing relapse risk by pattern
- Statistical significance indicators

**Key Insights to Look For**:
- Which patterns correlate with higher relapse rates
- Differences between CD3+ and CD3- pattern associations
- Sample sizes for each pattern category

### Phase 4: Data Preprocessing for ML

#### Step 4.1: Advanced Missing Value Imputation (Cell 10-11)
```python
# Sophisticated imputation and data cleaning
X_processed, y_classification, y_regression = prepare_ml_dataset(df_with_patterns)
```

**Processing Steps**:
1. **Patient Filtering**: AML/MDS patients only (`disease == 1`)
2. **Feature Selection**: Removes columns with >50% missing data
3. **Imputation**: Iterative imputation for moderate missing, median for high missing
4. **Data Alignment**: Ensures X and y indices match

**Expected Output**:
- Final dataset dimensions (e.g., 157 samples × 65 features)
- Zero missing values
- Balanced feature types summary

### Phase 5: Basic Machine Learning Analysis

#### Step 5.1: Feature Set Evaluation (Cell 12-15)
```python
# Tests different feature combinations
feature_sets = create_feature_sets_for_analysis()
all_results = evaluate_feature_set_performance(...)
```

**Feature Sets Tested**:
- **Dynamics Only**: Time-point differences
- **Time Points Only**: Raw measurements
- **Statistics Only**: Summary statistics
- **Patterns Only**: Trend classifications
- **Comprehensive**: All chimerism features
- **Minimal Predictive**: Top 3 features

**Expected Output**:
- Cross-validation accuracy for each set
- Best performing feature combinations per outcome
- Sample size and feature count summaries

#### Step 5.2: Model Training and Validation (Cell 16-17)
```python
# Trains optimized models with hyperparameter tuning
saved_models = train_and_save_optimized_models(X_ml, y_ml)
export_analysis_results(saved_models, X_ml, y_ml)
```

**Outputs**:
- Trained models saved to `models/` directory
- Model performance summaries
- CSV files with detailed results:
  - `basic_analysis_results_model_performance.csv`
  - `basic_analysis_results_feature_statistics.csv`
  - `basic_analysis_results_outcome_distributions.csv`
  - `basic_analysis_results_best_models.csv`

## Advanced Machine Learning Features

### Advanced Clustering Analysis

#### K-Means Clustering with UMAP (Cell 18)
```python
# Performs K-means clustering with 3D visualization
kmeans_results = perform_kmeans_clustering_with_umap(X_ml, n_clusters=4)
```

**Features**:
- **UMAP Dimensionality Reduction**: 3D projection for visualization
- **Interactive 3D Plots**: Plotly-based cluster visualization
- **Cluster Statistics**: Size, percentage, center coordinates
- **Silhouette Analysis**: Cluster quality metrics

**Expected Output**:
- Interactive 3D scatter plot
- Cluster summary statistics
- Cluster assignment labels

#### Fuzzy C-Means Clustering (Cell 19-20)
```python
# Performs fuzzy clustering with membership probabilities
fcm_results = perform_fcm_clustering_with_umap(X_ml, n_clusters=4, m=2.0)
```

**Features**:
- **Soft Assignments**: Membership probabilities for each cluster
- **Multiple Visualizations**: Hard assignments and membership strength
- **Fuzzy Metrics**: Fuzzy Partition Coefficient (FPC)
- **Membership Analysis**: Probability distributions and heatmaps

**Expected Output**:
- Two interactive 3D plots (hard + fuzzy assignments)
- Membership probability analysis plots
- FPC quality metric
- Membership probability heatmap

### Advanced Classification Analysis

#### Fuzzy SVM Classification (Cell 21-23)
```python
# Implements SVM with probability estimates
fuzzy_svm_results = implement_fuzzy_svm_classification(X_ml, y_ml[target_col], target_name)
```

**Features**:
- **Probability Estimates**: Class membership probabilities
- **Confidence Analysis**: Maximum probability distributions
- **Calibration Curves**: Probability calibration assessment
- **Multiple Visualizations**: Per-class and overall performance

**Expected Output**:
- Probability distribution plots for each class
- Confidence analysis histograms
- Calibration curves (for binary classification)
- Classification performance metrics

### Evolutionary Feature Selection

#### Genetic Algorithm Feature Selection (Cell 24-25)
```python
# Evolutionary feature selection with SVM evaluation
ga_results = implement_genetic_algorithm_feature_selection(
    X_ml, y_ml[target_col], target_name, 
    n_generations=30, population_size=30, k_folds=5
)
```

**Features**:
- **Evolutionary Optimization**: Binary encoding for feature selection
- **K-Fold Validation**: SVM accuracy as fitness function
- **Progress Tracking**: Fitness evolution over generations
- **Feature Importance**: Selection frequency analysis

**Expected Output**:
- Evolution progress plots
- Best feature subset identification
- Feature selection frequency analysis
- Performance comparison with full feature set

### Comprehensive Results Summary (Cell 26)
```python
# Creates unified summary of all advanced analyses
comprehensive_summary = create_comprehensive_results_summary()
```

**Features**:
- **Multi-Method Comparison**: Performance across all techniques
- **Statistical Summaries**: Key metrics for each method
- **Visualization Dashboard**: Comparative performance plots
- **CSV Export**: Detailed results for further analysis

## Output Files and Results

### Generated Files Structure
```
project_root/
├── models/                                    # Trained ML models
│   ├── best_{outcome}_k{n}_model.joblib     # Optimized models
│   └── y_{outcome}_Random_Forest_*.joblib   # Feature selection models
├── plots/                                    # Visualization outputs
│   ├── feat_imp_*.png                       # Feature importance plots
│   ├── roc_*.png                           # ROC curves
│   └── pr_*.png                            # Precision-recall curves
├── basic_analysis_results_*.csv             # Basic analysis results
├── advanced_analysis_*.csv                 # Advanced analysis results
└── basic_analysis_organized.ipynb          # Main notebook
```

### Key Result Files

#### 1. Model Performance Summary
**File**: `basic_analysis_results_model_performance.csv`
**Contents**:
- Target outcome
- Number of features used
- Cross-validation accuracy
- Best performing features
- Model file paths

#### 2. Feature Statistics
**File**: `basic_analysis_results_feature_statistics.csv`
**Contents**:
- Descriptive statistics for all engineered features
- Missing value percentages
- Feature correlation information

#### 3. Advanced Analysis Results
**Files**: 
- `advanced_analysis_fuzzy_svm_results.csv`
- `advanced_analysis_genetic_algorithm_results.csv`
- `advanced_analysis_clustering_results.csv`

**Contents**:
- Method-specific performance metrics
- Selected features and parameters
- Confidence and quality measures

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues
**Symptoms**: Kernel crashes, out-of-memory errors
**Solutions**:
- Reduce dataset size for testing: `df.sample(n=100)`
- Decrease GA population size: `population_size=20`
- Use fewer UMAP neighbors: `n_neighbors=10`

#### 2. Visualization Issues
**Symptoms**: Plotly plots not displaying, JavaScript errors
**Solutions**:
- Install/update Plotly: `pip install plotly>=5.0.0`
- Enable Jupyter extensions: `jupyter nbextension enable --py widgetsnbextension`
- Restart kernel and clear outputs

#### 3. Feature Engineering Errors
**Symptoms**: UFuncTypeError, data type issues
**Solutions**:
- Ensure numeric columns: Check data types with `df.dtypes`
- Handle missing values: Verify imputation completed successfully
- Convert object columns: Use `pd.to_numeric()` for numeric conversion

#### 4. Model Training Failures
**Symptoms**: Convergence warnings, poor performance
**Solutions**:
- Check target distribution: Ensure balanced classes
- Adjust model parameters: Increase max_iter, change C parameter
- Feature scaling: Verify StandardScaler applied correctly

#### 5. Advanced ML Component Failures
**Symptoms**: UMAP errors, scikit-fuzzy issues
**Solutions**:
- Check dependencies: Reinstall with exact versions
- Reduce complexity: Lower n_components, fewer clusters
- Data preprocessing: Ensure no infinite/NaN values

### Performance Optimization Tips

#### 1. For Large Datasets (>1000 samples)
- Use stratified sampling for initial exploration
- Implement batch processing for GA
- Consider PCA before UMAP for very high dimensions

#### 2. For Limited Compute Resources
- Reduce cross-validation folds to 3
- Use smaller GA populations (20-30)
- Limit UMAP iterations: `n_epochs=200`

#### 3. For Production Use
- Set fixed random seeds for reproducibility
- Implement proper train/validation/test splits
- Add comprehensive error logging

## Technical Details

### Algorithm Specifications

#### K-Means Clustering
- **Algorithm**: Lloyd's algorithm with kmeans++
- **Parameters**: n_clusters=4, n_init=10, random_state=42
- **Preprocessing**: StandardScaler normalization
- **Visualization**: UMAP to 3D (n_neighbors=15, min_dist=0.1)

#### Fuzzy C-Means Clustering
- **Algorithm**: Bezdek's FCM algorithm
- **Parameters**: m=2.0 (fuzziness), error=0.005, maxiter=1000
- **Quality Metric**: Fuzzy Partition Coefficient (FPC)
- **Membership**: Soft assignments with probability matrix

#### Fuzzy SVM Classification
- **Kernel**: RBF with gamma='scale'
- **Parameters**: C=1.0, probability=True
- **Evaluation**: Stratified train/test split (70/30)
- **Metrics**: Accuracy, precision, recall, F1-score

#### Genetic Algorithm Feature Selection
- **Encoding**: Binary chromosome (1=selected, 0=not selected)
- **Selection**: Tournament selection (tournsize=3)
- **Crossover**: Two-point crossover (prob=0.8)
- **Mutation**: Bit-flip mutation (prob=0.1)
- **Fitness**: K-fold SVM accuracy (k=5)

### Data Flow Architecture

```
Raw Data (CSV)
    ↓
Feature Engineering
    ↓ (25+ new features)
Pattern Classification
    ↓ (categorical patterns)
Data Preprocessing
    ↓ (imputation, scaling)
ML Analysis Pipeline
    ├── Basic ML (Random Forest + feature selection)
    ├── K-Means Clustering (+ UMAP visualization)
    ├── Fuzzy C-Means (+ membership analysis)
    ├── Fuzzy SVM (+ probability analysis)
    └── Genetic Algorithm (+ evolution tracking)
    ↓
Results Export (CSV + visualizations)
```

### Memory and Compute Requirements

#### Minimum Requirements
- **RAM**: 4GB
- **Dataset**: <500 samples, <100 features
- **Time**: ~10-15 minutes for full execution

#### Recommended Requirements
- **RAM**: 8GB+
- **Dataset**: 500-2000 samples, 100-200 features
- **Time**: ~20-30 minutes for full execution

#### Advanced Requirements (Large Scale)
- **RAM**: 16GB+
- **Dataset**: >2000 samples, >200 features
- **Time**: 1-2 hours for full execution
- **Storage**: 1GB+ for model outputs and visualizations

---

## Support and Contact

For technical issues or research questions:
1. Check the troubleshooting section above
2. Review the CLAUDE.md file for project-specific guidance
3. Examine log outputs for specific error messages
4. Consider reducing complexity for initial testing

## Version History

- **v1.0**: Initial implementation with basic ML pipeline
- **v1.1**: Added advanced clustering with UMAP visualization
- **v1.2**: Integrated fuzzy classification and genetic algorithms
- **v1.3**: Enhanced error handling and comprehensive documentation
- **v1.4**: Fixed visualization issues and improved CSV export

---

*This documentation is designed to provide complete guidance for executing the basic_analysis_organized.ipynb notebook. Each section builds upon the previous ones, creating a comprehensive machine learning pipeline for chimerism dynamics analysis in AML/MDS transplant research.*