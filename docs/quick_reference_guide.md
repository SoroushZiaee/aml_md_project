# Quick Reference Guide - Basic Analysis Organized Notebook

## 🚀 Quick Start Checklist

### Before You Begin
- [ ] Install required packages: `pip install -r requirements.txt`
- [ ] Ensure `preprocessed_ml_for_aml_mds.csv` is in project root
- [ ] Start Jupyter: `jupyter notebook` or `jupyter lab`
- [ ] Open `basic_analysis_organized.ipynb`

### Execution Order
1. **Setup** (Cells 1-3): Import libraries and load data
2. **Feature Engineering** (Cells 4-7): Create dynamic features and patterns
3. **Basic Analysis** (Cells 8-17): EDA and traditional ML
4. **Advanced Analysis** (Cells 18-26): Clustering, fuzzy methods, GA

## 📊 Key Functions Quick Reference

### Core Analysis Functions
```python
# Data loading and inspection
df = load_and_inspect_data("preprocessed_ml_for_aml_mds.csv")

# Feature engineering
df_enhanced = create_chimerism_dynamics_features(df)
df_with_patterns = create_pattern_features(df_enhanced)

# Data preprocessing
X_processed, y_classification, y_regression = prepare_ml_dataset(df_with_patterns)
```

### Advanced ML Functions
```python
# K-Means clustering with UMAP 3D
kmeans_results = perform_kmeans_clustering_with_umap(X_ml, n_clusters=4)

# Fuzzy C-Means clustering
fcm_results = perform_fcm_clustering_with_umap(X_ml, n_clusters=4, m=2.0)

# Fuzzy SVM classification
result = implement_fuzzy_svm_classification(X_ml, y_ml[target_col], target_name)

# Genetic algorithm feature selection
ga_results = implement_genetic_algorithm_feature_selection(
    X_ml, y_ml[target_col], target_name, 
    n_generations=30, population_size=30, k_folds=5
)
```

## 🎯 Expected Outputs by Section

### Section 1-3: Data Loading
- ✅ Dataset shape: ~250-300 rows × 60-70 columns
- ✅ Column summary with chimerism and outcome variables
- ✅ Memory usage under 1MB

### Section 4-7: Feature Engineering
- ✅ ~25-30 new features created
- ✅ Pattern distributions showing clinical trends
- ✅ Association analysis with outcome variables

### Section 8-11: Preprocessing
- ✅ Final ML dataset: ~150-200 samples × 60-70 features
- ✅ Zero missing values after imputation
- ✅ Standardized features ready for ML

### Section 12-17: Basic ML
- ✅ Feature set performance comparison
- ✅ Trained models saved to `models/` directory
- ✅ CSV results exported

### Section 18-20: Advanced Clustering
- ✅ Interactive 3D visualizations (Plotly)
- ✅ Cluster statistics and membership analysis
- ✅ UMAP dimensionality reduction plots

### Section 21-23: Fuzzy Classification
- ✅ Probability distribution plots
- ✅ Confidence analysis histograms
- ✅ Classification performance metrics

### Section 24-25: Genetic Algorithm
- ✅ Evolution progress visualization
- ✅ Feature selection frequency analysis
- ✅ Optimized feature subsets identified

### Section 26: Comprehensive Summary
- ✅ Multi-method performance comparison
- ✅ Advanced analysis CSV exports
- ✅ Final visualization dashboard

## ⚠️ Common Issues & Quick Fixes

### Import Errors
```bash
# Missing packages
pip install umap-learn==0.5.3 scikit-fuzzy==0.4.2 deap==1.3.3

# Plotly visualization issues
pip install plotly>=5.0.0
jupyter nbextension enable --py widgetsnbextension
```

### Data Issues
```python
# File not found
# Ensure preprocessed_ml_for_aml_mds.csv is in project root

# Memory issues with large datasets
df_sample = df.sample(n=100)  # Use smaller sample for testing
```

### Visualization Issues
```python
# Plotly not displaying
# Restart kernel and clear all outputs
# Try: Kernel > Restart & Clear Output

# UMAP taking too long
umap_3d = umap.UMAP(n_components=3, n_neighbors=10, n_epochs=100)
```

### Performance Issues
```python
# Reduce GA complexity
ga_results = implement_genetic_algorithm_feature_selection(
    X_ml, y_ml[target_col], target_name,
    n_generations=20,  # Reduced from 30
    population_size=20,  # Reduced from 30
    k_folds=3  # Reduced from 5
)
```

## 📁 Output Files Directory Structure

```
project_root/
├── models/
│   ├── best_relapse_k1_model.joblib
│   ├── best_death_k2_model.joblib
│   └── ...
├── basic_analysis_results_model_performance.csv
├── basic_analysis_results_feature_statistics.csv
├── advanced_analysis_fuzzy_svm_results.csv
├── advanced_analysis_genetic_algorithm_results.csv
└── advanced_analysis_clustering_results.csv
```

## 🎛️ Parameter Tuning Guide

### For Better Performance
```python
# Clustering
n_clusters = 3  # Try 3-6 clusters
n_neighbors = 10  # UMAP: try 5-20

# Fuzzy C-Means
m = 1.5  # Fuzziness: 1.1-3.0 (lower = crisper)

# SVM
C = 0.1  # Regularization: 0.01-10
gamma = 'auto'  # or try 0.001-1.0

# Genetic Algorithm
n_generations = 50  # More generations = better solutions
population_size = 50  # Larger = more diverse search
```

### For Faster Execution
```python
# Reduce complexity
n_clusters = 3
n_generations = 20
population_size = 20
k_folds = 3
n_neighbors = 10  # UMAP
```

## 🔍 Key Metrics to Monitor

### Data Quality
- Missing values percentage < 50%
- Sample size > 50 per outcome
- Feature correlation < 0.95

### Model Performance
- Cross-validation accuracy > 0.6
- Feature importance > 0.01
- Confidence scores > 0.7

### Advanced Methods
- FPC (Fuzzy Partition Coefficient) > 0.7
- GA fitness improvement > 0.05
- UMAP neighborhood preservation > 0.8

## 📞 Quick Help Commands

```python
# Check data status
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Column types: {df.dtypes.value_counts()}")

# Check ML readiness
print(f"ML features: {X_ml.shape}")
print(f"ML targets: {y_ml.shape}")
print(f"Target distribution: {y_ml.sum()}")

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
```

---

**💡 Pro Tip**: Start with a small sample (n=50-100) to test all functions work correctly, then run on full dataset. This saves time during development and debugging.

**🔄 Recommended Workflow**: Execute cells 1-17 first (basic analysis), then proceed to advanced features (18-26) based on your research needs.

**📈 Performance Note**: Full execution takes 20-45 minutes depending on dataset size and hardware. Advanced ML sections (18-26) are the most computationally intensive.