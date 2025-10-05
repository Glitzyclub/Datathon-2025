# NSP Classification of UCI CTG Dataset

## 🏆 Hackathon Project: Fetal Health Classification Using XGBoost

A machine learning solution for automated fetal health assessment using cardiotocography (CTG) data, achieving **93-95% classification accuracy** through advanced gradient boosting techniques.

## 📋 Project Overview

This project implements an automated classification system for fetal health status using the UCI Cardiotocography dataset. The system classifies fetal conditions into three categories:

- **Normal (N)**: Healthy fetal conditions
- **Suspect (S)**: Conditions requiring monitoring
- **Pathological (P)**: Critical conditions requiring immediate intervention

## 🎯 Key Achievements

- **93-95% Classification Accuracy**
- **Robust handling of class imbalance** (9.4:1 ratio)
- **Bayesian optimization** for hyperparameter tuning
- **Clinical-grade performance** suitable for medical decision support

## 📊 Dataset Information

- **Source**: UCI Machine Learning Repository
- **Samples**: 2,126 cardiotocography records
- **Features**: 21 numerical features
- **Classes**: 3 (Normal: 77.8%, Suspect: 13.9%, Pathological: 8.3%)

### Feature Categories
- **Baseline Measurements**: LB (Baseline FHR), UC (Uterine Contractions)
- **Variability Measures**: ASTV, MSTV, ALTV, MLTV
- **Acceleration/Deceleration**: AC, DL, DS, DP
- **Histogram Features**: Width, Min, Max, Mode, Mean, Median, Variance
- **Morphological Patterns**: Tendency, Nmax, Nzeros

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Required Libraries
```bash
pip install pandas numpy scikit-learn xgboost scikit-optimize matplotlib seaborn scipy
```

## 🚀 Usage

### Quick Start
```python
# Load and run the complete pipeline
python main.py

# For individual components:
python data_exploration.py    # Data analysis and visualization
python model_training.py      # XGBoost training with Bayesian optimization
python evaluation.py          # Model evaluation and metrics
```

### Custom Configuration
```python
from src.model import XGBoostClassifier
from src.optimization import BayesianOptimizer

# Initialize classifier with custom parameters
classifier = XGBoostClassifier(
    use_class_weights=True,
    n_estimators=100,
    max_depth=6
)

# Run Bayesian optimization
optimizer = BayesianOptimizer(classifier)
best_params = optimizer.optimize(X_train, y_train, n_calls=50)
```

## 🔬 Methodology

### 1. Data Exploration
- **Class Distribution Analysis**: Identified severe class imbalance
- **Feature Distribution Analysis**: Examined statistical properties
- **Correlation Matrix**: Assessed feature relationships
- **ANOVA F-test**: Statistical significance testing for feature selection

### 2. Model Architecture
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Class Imbalance Handling**: Automatic class weighting
- **Feature Processing**: No scaling (leveraging tree-based algorithm properties)
- **Hyperparameter Optimization**: Bayesian optimization with Gaussian processes

### 3. Key Technical Decisions
- **Unscaled Data**: Maintained clinical interpretability
- **Class Weights**: Addressed imbalance without synthetic data
- **Bayesian Optimization**: Efficient hyperparameter search
- **Cross-validation**: Robust performance estimation

## 📈 Results

### Performance Metrics
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 95.0% | 97.0% | 96.0% |
| Suspect | 87.0% | 83.0% | 85.0% |
| Pathological | 91.0% | 89.0% | 90.0% |

- **Overall Accuracy**: 94.5%
- **ROC-AUC**: 92.0%
- **Performance Range**: 93-95% across validation folds

### Comparison with Literature
- **This Study**: 93-95% (XGBoost + Bayesian Optimization)
- **Chen et al. (2022)**: 97.6% (Ada-RF ensemble)
- **Kuzu et al. (2023)**: 94-96% (Deep Learning)
- **Traditional Methods**: 90-92% (baseline)

## 📁 Project Structure

```
├── data/
│   ├── raw/                 # Original UCI CTG dataset
│   ├── processed/           # Cleaned and preprocessed data
│   └── results/             # Model outputs and predictions
├── src/
│   ├── data_exploration.py  # EDA and statistical analysis
│   ├── preprocessing.py     # Data preparation utilities
│   ├── model.py            # XGBoost classifier implementation
│   ├── optimization.py     # Bayesian optimization framework
│   └── evaluation.py       # Performance metrics and validation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_results_analysis.ipynb
├── results/
│   ├── figures/            # Plots and visualizations
│   ├── models/             # Saved model artifacts
│   └── reports/            # Performance reports
├── tests/                  # Unit tests
├── requirements.txt
├── main.py                 # Main execution script
└── README.md
```

## 🔍 Key Features

### Advanced Techniques
- **Bayesian Optimization**: Intelligent hyperparameter search using Gaussian processes
- **Class Weighting**: Automatic handling of imbalanced medical data
- **Statistical Validation**: ANOVA F-test for feature significance
- **Cross-validation**: Robust performance estimation

### Clinical Relevance
- **Medical Interpretability**: Preserved original feature scales
- **Critical Case Detection**: High sensitivity for pathological conditions
- **Real-world Applicability**: Handles typical medical data characteristics

## 🏥 Clinical Applications

This model can serve as a **clinical decision support tool** for:
- Fetal health monitoring during pregnancy
- Early detection of concerning conditions
- Risk stratification in obstetric care
- Supporting clinical decision-making in resource-limited settings

## 📊 Visualizations

The project includes comprehensive visualizations:
- Class distribution analysis
- Feature correlation heatmaps
- Performance metric comparisons
- ROC curves and confusion matrices
- Bayesian optimization convergence plots


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. UCI Machine Learning Repository - Cardiotocography Dataset
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system
3. Mockus, J. (2012). Bayesian approach to global optimization
4. Clinical studies on cardiotocography and fetal monitoring
---

⭐ **Star this repository if you found it helpful!**

**Keywords**: `machine-learning` `healthcare-ai` `xgboost` `bayesian-optimization` `medical-classification` `fetal-health` `cardiotocography` `imbalanced-data` `hackathon-project`