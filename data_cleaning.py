# ============================================================
# Task 2: Data Cleaning & Missing Value Handling
# Edutech Solution – AI & ML Internship
# Dataset: Titanic (891 rows × 12 columns)
# Tools: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("   TASK 2: DATA CLEANING & MISSING VALUE HANDLING")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("\n📂 STEP 1: Load Dataset")
print("─" * 65)

df = pd.read_csv("titanic.csv")
print(f"   Dataset : Titanic")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")
print(f"\n   First 5 rows:")
print(df.head().to_string())

# ─────────────────────────────────────────────────────────────
# STEP 2 — IDENTIFY MISSING VALUES
# ─────────────────────────────────────────────────────────────
print("\n\n❓ STEP 2: Identify Missing Values")
print("─" * 65)

missing_count = df.isnull().sum()
missing_pct   = (missing_count / len(df) * 100).round(2)
missing_df    = pd.DataFrame({
    "Missing Count" : missing_count,
    "Missing (%)"   : missing_pct,
    "Data Type"     : df.dtypes
})
missing_with = missing_df[missing_df["Missing Count"] > 0]

print(f"\n   Total missing cells : {df.isnull().sum().sum()}")
print(f"   Columns with missing: {len(missing_with)}\n")
print(missing_with.to_string())

# ─────────────────────────────────────────────────────────────
# STEP 3 — VISUALIZE MISSING DATA (Heatmap)
# ─────────────────────────────────────────────────────────────
print("\n\n🗺️  STEP 3: Visualize Missing Data (Heatmap saved as PNG)")
print("─" * 65)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Titanic Dataset – Missing Value Analysis", fontsize=16, fontweight='bold')

# Heatmap
sns.heatmap(df.isnull(), yticklabels=False, cbar=True,
            cmap='viridis', ax=axes[0])
axes[0].set_title("Missing Value Heatmap\n(Yellow = Missing)", fontsize=13)
axes[0].set_xlabel("Columns")

# Bar chart of missing %
missing_with["Missing (%)"].plot(kind="bar", color=["#e74c3c","#e67e22","#9b59b6"],
                                  ax=axes[1], edgecolor='black')
axes[1].set_title("Missing Value % per Column", fontsize=13)
axes[1].set_ylabel("Missing (%)")
axes[1].set_xticklabels(missing_with.index, rotation=15)
for i, v in enumerate(missing_with["Missing (%)"]):
    axes[1].text(i, v + 0.5, f"{v}%", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("missing_value_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: missing_value_heatmap.png")

# ─────────────────────────────────────────────────────────────
# STEP 4 — TYPES OF MISSING DATA
# ─────────────────────────────────────────────────────────────
print("\n\n📖 STEP 4: Types of Missing Data (MCAR / MAR / MNAR)")
print("─" * 65)
print("""
   Age      → MAR  (Missing At Random) – younger passengers less likely
               to have age recorded; correlates with Pclass/Sex
   Cabin    → MNAR (Missing Not At Random) – mostly lower-class passengers
               didn't have cabin assignments; value relates to missingness
   Embarked → MCAR (Missing Completely At Random) – only 2 rows missing,
               no apparent pattern
""")

# ─────────────────────────────────────────────────────────────
# STEP 5 — HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────
print("\n🔧 STEP 5: Handle Missing Values")
print("─" * 65)

df_clean = df.copy()

# 5a — Drop 'Cabin' (>77% missing — too much to impute reliably)
print("\n   [A] DROP 'Cabin' column — 77.1% missing (unreliable to impute)")
df_clean.drop(columns=["Cabin"], inplace=True)
print(f"       Columns remaining: {df_clean.shape[1]}")

# 5b — Impute 'Age' with Median (skewed distribution, robust to outliers)
print("\n   [B] IMPUTE 'Age' with MEDIAN — avoids outlier bias")
age_median = df_clean["Age"].median()
print(f"       Median Age = {age_median}")
df_clean["Age"] = df_clean["Age"].fillna(age_median)
print(f"       Missing 'Age' after imputation: {df_clean['Age'].isnull().sum()}")

# 5c — Impute 'Embarked' with Mode (categorical, only 2 missing)
print("\n   [C] IMPUTE 'Embarked' with MODE — categorical, only 2 missing")
embarked_mode = df_clean["Embarked"].mode()[0]
print(f"       Mode Embarked = '{embarked_mode}'")
df_clean["Embarked"] = df_clean["Embarked"].fillna(embarked_mode)
print(f"       Missing 'Embarked' after imputation: {df_clean['Embarked'].isnull().sum()}")

# 5d — Impute 'Fare' with Median (if any missing)
if df_clean["Fare"].isnull().sum() > 0:
    df_clean["Fare"].fillna(df_clean["Fare"].median(), inplace=True)
    print("\n   [D] IMPUTE 'Fare' with MEDIAN")

print("\n   ✅ Missing values handled successfully!")

# ─────────────────────────────────────────────────────────────
# STEP 6 — SIMPLEIMPUTER (Scikit-learn)
# ─────────────────────────────────────────────────────────────
print("\n\n🤖 STEP 6: SimpleImputer from Scikit-learn (Demonstration)")
print("─" * 65)

df_si = df[["Age", "Fare"]].copy()
print(f"   Before imputation — Age missing: {df_si['Age'].isnull().sum()}, "
      f"Fare missing: {df_si['Fare'].isnull().sum()}")

imputer = SimpleImputer(strategy="median")
df_si[["Age", "Fare"]] = imputer.fit_transform(df_si[["Age", "Fare"]])

print(f"   After  imputation — Age missing: {df_si['Age'].isnull().sum()}, "
      f"Fare missing: {df_si['Fare'].isnull().sum()}")
print(f"   Strategy used: median")
print(f"   Imputed statistics: Age={imputer.statistics_[0]:.1f}, Fare={imputer.statistics_[1]:.2f}")

# ─────────────────────────────────────────────────────────────
# STEP 7 — OUTLIER DETECTION & TREATMENT (IQR Method)
# ─────────────────────────────────────────────────────────────
print("\n\n📦 STEP 7: Outlier Detection & Treatment (IQR Method)")
print("─" * 65)

def detect_outliers_iqr(series):
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = series[(series < lower) | (series > upper)]
    return outliers, lower, upper

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Outlier Detection – Before & After Treatment", fontsize=15, fontweight='bold')

outlier_cols = ["Age", "Fare", "SibSp", "Parch"]

for i, col in enumerate(outlier_cols):
    ax = axes[i // 2][i % 2]
    outliers, lower, upper = detect_outliers_iqr(df_clean[col])
    print(f"\n   '{col}': {len(outliers)} outliers detected "
          f"(bounds: {lower:.2f} – {upper:.2f})")

    # Cap outliers (Winsorization)
    df_clean[col] = np.clip(df_clean[col], lower, upper)

    ax.boxplot(df_clean[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.7))
    ax.set_title(f"'{col}' after Winsorization", fontsize=11)
    ax.set_ylabel("Value")

plt.tight_layout()
plt.savefig("outlier_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n   ✅ Saved: outlier_boxplots.png")
print("   ✅ Outliers capped using IQR Winsorization")

# ─────────────────────────────────────────────────────────────
# STEP 8 — ADDITIONAL CLEANING
# ─────────────────────────────────────────────────────────────
print("\n\n🧹 STEP 8: Additional Cleaning Steps")
print("─" * 65)

# Drop irrelevant columns
df_clean.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
print(f"   Dropped: PassengerId, Name, Ticket (non-informative for ML)")

# Encode categorical columns
df_clean["Sex"]      = df_clean["Sex"].map({"male": 0, "female": 1})
df_clean["Embarked"] = df_clean["Embarked"].map({"S": 0, "C": 1, "Q": 2})
print(f"   Encoded: Sex (male=0, female=1)")
print(f"   Encoded: Embarked (S=0, C=1, Q=2)")

# Check duplicates
dupes = df_clean.duplicated().sum()
if dupes > 0:
    df_clean.drop_duplicates(inplace=True)
print(f"   Duplicate rows removed: {dupes}")

print(f"\n   Final shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# STEP 9 — VERIFY CLEAN DATASET
# ─────────────────────────────────────────────────────────────
print("\n\n✅ STEP 9: Verify Clean Dataset")
print("─" * 65)

final_missing = df_clean.isnull().sum().sum()
print(f"   Total missing cells   : {final_missing}")
print(f"   Duplicate rows        : {df_clean.duplicated().sum()}")
print(f"   Final shape           : {df_clean.shape}")
print(f"\n   Data types after cleaning:")
print(df_clean.dtypes.to_string())
print(f"\n   Summary statistics:")
print(df_clean.describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────
# STEP 10 — COMPARISON VISUALIZATION
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Age Distribution – Before vs After Cleaning", fontsize=14, fontweight='bold')

df["Age"].dropna().hist(bins=30, color='#e74c3c', alpha=0.7, ax=axes[0], edgecolor='black')
axes[0].set_title("Before Cleaning (with missing + outliers)", fontsize=11)
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")

df_clean["Age"].hist(bins=30, color='#2ecc71', alpha=0.7, ax=axes[1], edgecolor='black')
axes[1].set_title("After Cleaning (imputed + capped)", fontsize=11)
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("age_before_after.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n   ✅ Saved: age_before_after.png")

# ─────────────────────────────────────────────────────────────
# SAVE CLEANED CSV
# ─────────────────────────────────────────────────────────────
df_clean.to_csv("titanic_cleaned.csv", index=False)
print(f"\n   ✅ Saved: titanic_cleaned.csv ({df_clean.shape[0]} rows × {df_clean.shape[1]} cols)")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("   CLEANING SUMMARY REPORT")
print("=" * 65)
print(f"""
   Original Dataset : 891 rows × 12 columns
   Cleaned Dataset  : {df_clean.shape[0]} rows × {df_clean.shape[1]} columns

   Actions Taken:
     • Dropped  'Cabin'     → 77.1% missing (unreliable)
     • Dropped  'PassengerId', 'Name', 'Ticket' → non-informative
     • Imputed  'Age'       → Median imputation
     • Imputed  'Embarked'  → Mode imputation
     • Treated  Outliers    → IQR Winsorization on Age, Fare, SibSp, Parch
     • Encoded  'Sex'       → Binary (0/1)
     • Encoded  'Embarked'  → Ordinal (0/1/2)

   Missing Values    : {final_missing} (was {df.isnull().sum().sum()})
   Duplicate Rows    : {df_clean.duplicated().sum()}

   Output Files:
     ✅ titanic_cleaned.csv
     ✅ missing_value_heatmap.png
     ✅ outlier_boxplots.png
     ✅ age_before_after.png
""")
print("=" * 65)
print("  ✅ Task 2 Data Cleaning Complete!")
print("=" * 65)
