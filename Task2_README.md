# 🧹 Task 2: Data Cleaning & Missing Value Handling

### Edutech Solution – AI & ML Internship

---

## 🎯 Objective

Master the art of **data preprocessing and quality assurance** by cleaning the Titanic dataset — handling missing values, treating outliers, and producing a fully clean, ML-ready dataset.

---

## 📁 Repository Structure

```
├── titanic.csv                     # Original dataset (891 rows × 12 cols)
├── titanic_cleaned.csv             # ✅ Cleaned output (757 rows × 8 cols)
├── data_cleaning.py                # Full Python cleaning script
├── Task2_Data_Cleaning.ipynb       # Jupyter Notebook with all steps
├── missing_value_heatmap.png       # Visualization – missing data
├── outlier_boxplots.png            # Visualization – outlier detection
├── age_before_after.png            # Visualization – before vs after
└── README.md                       # This file
```

---

## 🗂️ Dataset: Titanic

| Column | Type | Missing | Action Taken |
|--------|------|---------|--------------|
| PassengerId | int64 | 0 | Dropped (non-informative) |
| Survived | int64 | 0 | Kept (target variable) |
| Pclass | int64 | 0 | Kept |
| Name | object | 0 | Dropped (non-informative) |
| Sex | object | 0 | Encoded → 0/1 |
| Age | float64 | 177 (19.9%) | **Median Imputation** |
| SibSp | int64 | 0 | Kept + Winsorized |
| Parch | int64 | 0 | Kept + Winsorized |
| Ticket | object | 0 | Dropped (non-informative) |
| Fare | float64 | 0 | Kept + Winsorized |
| Cabin | object | 687 (77.1%) | **Dropped** |
| Embarked | object | 2 (0.2%) | **Mode Imputation** |

---

## ✅ Tasks Completed

| # | Task | Method | Status |
|---|------|--------|--------|
| 1 | Identify missing values | `.isnull().sum()` | ✅ |
| 2 | Visualize missing data | Seaborn Heatmap | ✅ |
| 3 | Handle missing — Deletion | `dropna()` / `drop()` | ✅ |
| 4 | Handle missing — Imputation | Mean / Median / Mode | ✅ |
| 5 | Automated imputation | `SimpleImputer` (Scikit-learn) | ✅ |
| 6 | Treat outliers | IQR Winsorization | ✅ |
| 7 | Verify clean dataset | `.isnull().sum()` = 0 | ✅ |

---

## 📊 Cleaning Results

| Metric | Before | After |
|--------|--------|-------|
| Rows | 891 | 757 |
| Columns | 12 | 8 |
| Missing Cells | 866 | **0** |
| Duplicate Rows | 122 | **0** |
| Outliers | Present | Capped (IQR) |

---

## 🔍 Types of Missing Data Found

- **Age → MAR** (Missing At Random) — correlates with Pclass/Sex
- **Cabin → MNAR** (Missing Not At Random) — lower class had no cabin
- **Embarked → MCAR** (Missing Completely At Random) — 2 random rows

---

## 🛠️ Tools & Technologies

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **IDE:** Jupyter Notebook / VS Code

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/vishnubabalsure/Task2-Data-Cleaning.git
cd Task2-Data-Cleaning

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Run Python script
python data_cleaning.py

# OR open Jupyter Notebook
jupyter notebook Task2_Data_Cleaning.ipynb
```

---

## 💡 Interview Q&A

**Q1. Types of missing data (MCAR, MAR, MNAR)?**  
MCAR = random, no pattern. MAR = relates to other observed variables. MNAR = value itself causes missingness.

**Q2. When to drop vs impute?**  
Drop when >50% missing or column is irrelevant. Impute when the feature is important and data is MCAR/MAR.

**Q3. How do outliers affect imputation?**  
Outliers skew the Mean upward/downward, making mean-imputation unreliable. Median is robust to outliers.

**Q4. Mean vs Median imputation?**  
Mean for symmetric distributions. Median for skewed distributions or when outliers are present (like Age, Fare).

**Q5. Role of data cleaning in ML pipeline?**  
Ensures model trains on accurate data — prevents garbage-in/garbage-out. Directly impacts model accuracy, convergence, and fairness.

---

*Submitted as part of Edutech Solution AI & ML Internship – Task 2*
