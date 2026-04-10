# 🍄 Mushroom Classification — EDA & Random Forest with Pipeline

A machine learning project that classifies mushrooms as **edible or poisonous** based on their physical characteristics, using the UCI Mushroom Dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Results](#results)
- [License](#license)

---

## Overview

This notebook covers the full machine learning workflow, from exploratory data analysis to model deployment via a sklearn Pipeline :

- **Exploratory Data Analysis** — class distribution, feature count plots, and correlation analysis using Cramér's V
- **Preprocessing** — LabelEncoder for the target variable, OneHotEncoder for categorical features, with careful attention to avoiding data leakage by splitting before encoding
- **Model comparison** — Decision Tree, Gradient Boosting, AdaBoost, Random Forest, Bagging, and MLP evaluated with StratifiedKFold cross-validation
- **Hyperparameter optimization** — RandomizedSearchCV on the Random Forest model
- **Model interpretability** — feature importances and decision tree visualization
- **Prediction on a new sample** — with probability scores and a nearest neighbor confidence check
- **Pipeline implementation** — replicating the full workflow using sklearn's Pipeline to automate preprocessing and secure against data leakage

---

## Dataset

**Source :** [UCI Machine Learning Repository — Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom)

| Property | Value |
|---|---|
| Samples | 8124 |
| Features | 22 categorical |
| Target | edible (e) / poisonous (p) |
| Missing values | None |

The dataset describes mushroom samples from 23 species of gilled mushrooms. Each sample is described by physical characteristics such as cap shape, odor, gill color, spore print color, and habitat.

---

## Project Structure

```
📦 mushroom-classification
├── 📓 Mushrooms.ipynb       # Main notebook
├── 📁 data/
│   └── Mushrooms.csv        # Dataset
└── 📄 README.md
```

---

## Installation

**Clone the repository**
```bash
git clone https://github.com/Mattys565/mushroom-classification.git
cd mushroom-classification
```

**Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

**Run the notebook**
```bash
jupyter notebook Mushrooms.ipynb
```

---

## Results

All six tested models achieved near-perfect accuracy on this dataset, confirming that it is linearly separable. The **Random Forest** was selected for its robustness, interpretability, and zero misclassification of poisonous mushrooms.

| Model | Accuracy | F1 Score |
|---|---|---|
| Decision Tree | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 |
| Gradient Boosting | 1.00 | 1.00 |
| AdaBoost | 1.00 | 1.00 |
| Bagging | 1.00 | 1.00 |
| MLP | 1.00 | 1.00 |

After hyperparameter optimization with `RandomizedSearchCV`, the best parameters found were :

```
n_estimators     : 70
criterion        : log_loss
max_depth        : 30
min_samples_split: 25
min_samples_leaf : 1
max_features     : log2
```

### Key Findings

The most important features for predicting mushroom toxicity were :

- **odor** (none and foul) — the most discriminating feature
- **gill-color** (buff)
- **gill-size** (broad and narrow)
- **spore-print-color** (chocolate)

A mushroom with no odor and buff gills is very likely edible. A foul-smelling mushroom with a chocolate spore print is a strong indicator of toxicity.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
