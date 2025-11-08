# 📘 Advanced Regression Project(MDTS4313): Exploring Nonlinear Relationships in Global Health Data

## 🧩 Project Overview
This project investigates the **nonlinear relationship between per-capita health expenditure and life expectancy** using advanced **nonparametric regression techniques**. The analysis leverages data from the **World Health Organization (WHO)** and **United Nations (2000–2015)** to empirically validate the **economic principle of diminishing marginal returns** — where life expectancy gains taper off as health spending increases.

The workflow includes:
- Rigorous **data cleaning and exploratory data analysis (EDA)**
- Implementation of **six nonparametric smoothers**
- **5-fold cross-validation** for hyperparameter tuning
- Comparative evaluation using **Mean Squared Error (MSE)**

---

## 📊 Dataset Information
**Source:** [WHO Life Expectancy Dataset (Kaggle)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

- **Observations:** 2938 (193 countries, 2000–2015)
- **Independent Variable (X):** `percentage_expenditure` → Health expenditure per capita  
- **Dependent Variable (Y):** `life_expectancy` → Life expectancy at birth (years)

After cleaning:
- Removed **10 missing-value records**
- Excluded **32 outliers** (`X > 99th percentile` or `Y < 40`)
- Final dataset size: **2896 observations**

---

## 🧮 Methodology

### 1️⃣ Data Preparation
- Standardized column names
- Handled missing and extreme values
- Visualized distributions and relationships
- Detected and removed outliers using percentile thresholds

### 2️⃣ Exploratory Data Analysis (EDA)
- **Global Statistics:** Identified large disparities between developed and developing nations  
- **Scatterplots:** Revealed strong, saturating nonlinear patterns  
- **Time Series (2000–2015):** Confirmed spending-life expectancy linkage  
- **Heatmaps:** Showed structured missingness across select variables  

### 3️⃣ Modeling Approach
Each smoother was implemented in **Python (scikit-learn & statsmodels)** with 5-fold cross-validation on the training set (80%) and tested on a 20% hold-out set.

**Implemented Methods:**
| Method | Key Hyperparameter | Optimal Value | Test MSE |
|--------|--------------------|---------------|-----------|
| KNN (Uniform Weights) | `k` | 100 | 66.73 |
| KNN (Distance Weights) | `k` | 200 | 73.74 |
| Bin Smoother | `n_bins` | 75 | 67.62 |
| LOWESS (statsmodels) | `frac` | 0.3 | **66.59** |
| Kernel Smoother (Gaussian) | `h` | 15 | 68.23 |
| Local Linear Regression (Tricube) | `h` | 200 | 69.48 |

---

## 🧠 Key Findings
- The **LOWESS model (frac=0.3)** achieved the **lowest test MSE = 66.59**, providing the smoothest and most generalizable fit.  
- All smoothers captured the **“diminishing returns”** phenomenon — rapid life expectancy increases at low expenditures and a plateau at high expenditures.  
- Hyperparameter tuning was critical for balancing **bias–variance tradeoff**.  
- Models like **KNN (Distance)** and **Kernel Smoother** were more flexible but prone to overfitting, while **Bin Smoother** and **LOWESS** offered robust generalization.

---

## 💡 Insights & Policy Implications
- **For low-spending countries:** Increasing healthcare investment yields large life expectancy gains.  
- **For high-spending countries:** Further gains are minimal; focus should shift to efficiency and socio-economic determinants.  
- Demonstrates that **nonparametric regression** effectively uncovers structural nonlinearities that linear models would miss.

---

## ⚙️ Technical Details
**Languages & Libraries:**
```python
Python, pandas, numpy, matplotlib, seaborn, sklearn, statsmodels
```
## Core Functions:
- BinSmoother() (custom implementation)
- kernel_smoother() for Nadaraya–Watson regression
- locally_weighted_regression() for Local Linear Regression
- plot_cv_results() for visualizing cross-validation curves

## Validation Setup:
- 5-Fold Cross-Validation (k=5)
- Metric: Mean Squared Error (MSE)
- 80:20 Train-Test Split
- Random State: 42 (for reproducing same results)
---
  
## 🧭 Conclusions
- LOWESS (frac=0.3) emerged as the optimal smoother, achieving the lowest error and best curve shape.
- Cross-validation played a vital role in controlling model complexity.
- The project demonstrates how hyperparameter tuning is essential for optimizing nonparametric regression performance.
- Future extensions may include multivariate models such as Generalized Additive Models (GAMs) to capture multi-factor interactions.
  
## 📦 Repository Structure
```css
life-expectancy-smoothing-analysis/
│
├── requirements.txt
├── Life Expectancy Data.csv
├── code/
│   ├── code.ipynb
│   ├── smoothers.py   #core logic & funcs
│   └── figures/
│       ├── cv_plot_lowess.png
│       ├── cv_plot_bin_smoother.png
│       ├── .......    #all other .png files
│       ├── eda_scatterplot_all_by_status.png
│       └── model_comparison_plot.png
├── report/
│   └── MDTS4313_Sannidhya_419.pdf
└── README.md

```
---
## 🔗 References
- Dataset: [WHO Life Expectancy (Kaggle)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- Course: Advanced Regression (MDTS4313)
- Author: Sannidhya Das (Roll No. 419, MSc Semester 3)
- Date: November 2025
```kotlin
👨‍💻 Developed and maintained by: Sannidhya Das  
📬 For academic inquiries: [sannidhyadas0howrah@gmail.com]
```
