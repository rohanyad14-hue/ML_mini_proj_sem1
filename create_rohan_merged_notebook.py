import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

nb = new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.11"
    }
}

cells = []

# --- CELL 1: Header Markdown ---
cells.append(new_markdown_cell("""# Vellore Institute of Technology (VIT)
## School of Computer Science and Engineering
### M.Tech Course: Machine Learning Lab (MACSE504)
### Case Study Based Mini-Project — Phase 1: Part A (10 Marks)

---

| **Student Name** | **Register Number** | **Assigned Question** | **Evaluation Component** | **Submission Deadline** |
| :--- | :--- | :--- | :--- | :--- |
| **Rohan** | **26MML0020** | **Question 1: Predictive Churn Risk Modeling for Telecom Subscription Services** | **Part A: Exploratory Data Analysis & Preprocessing** | **5th October 2026** |

---

### **Executive Summary & Problem Statement**
* **Industry Case Scenario**: A telecommunications enterprise aims to proactively reduce subscriber churn by identifying at-risk customers early. The dataset encapsulates multi-faceted subscriber profiles across demographics, subscribed services, account administration, and billing attributes.
* **Dataset Reference**: Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
* **Scope of Merged Deliverables**:
  1. **Dataset Ingestion & Deep Structural Profiling**: Ingest dataset, inspect shape, summary statistics, duplicate records, missing values, column data types, and unique values audit.
  2. **TotalCharges Parsing & Anomaly Treatment**: Detect whitespace strings (`" "`) in `TotalCharges`, link them to $0$-tenure customer accounts, drop irrelevant identifier `customerID`, parse into `float64`, and clean missing values.
  3. **Mandatory Part A Question 1 Deliverables**:
     - Quantify churn proportions and conduct $\chi^2$ hypothesis tests across **Contract Types** (*Month-to-month, One year, Two year*) and **Internet Service Types** (*DSL, Fiber optic, No*).
     - Generate distribution plots of **Tenure vs. Monthly Charges**, bivariate density contours, and correlation matrices for continuous numerical attributes.
  4. **Domain-Specific Feature Engineering**:
     - `TotalServices`: Aggregated count of subscribed telecom services.
     - `LifetimeValue`: Proxy metric computed as $\text{MonthlyCharges} \times \text{tenure}$.
     - `AvgMonthlySpend`: Empirical monthly run-rate $\frac{\text{TotalCharges}}{\text{tenure}}$.
     - `HasInternet`: Binary indicator for internet access.
     - `ProtectionScore`: Security bundle score spanning $\text{OnlineSecurity} + \text{DeviceProtection} + \text{TechSupport}$.
  5. **Encoding & Export**:
     - Binary mapping for two-class features (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`).
     - Correlation heatmap of all numerical and binary features.
     - One-Hot Encoding for multi-categorical predictors with `drop_first=True, dtype=int` yielding $(7032, 36)$ clean matrix.
     - Export finalized dataset as `telco_preprocessed.csv`.
  6. **Interactive GUI Widget Console**: Dynamic interface for dataset loading, parameter tuning, cohort filtering, and visualization (rubric compliance).
"""))

# --- CELL 2: Environment Configuration & Library Ingestion ---
cells.append(new_code_cell("""# ==========================================================
# IMPORT LIBRARIES & CONFIGURE ENVIRONMENT
# ==========================================================
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Numerical & Data Manipulation
import numpy as np
import pandas as pd
import scipy.stats as stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & Model Selection utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)

# Interactive Widgets for GUI Interface (Rubric Requirement)
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Configuration for aesthetic publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 120,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8
})

CHURN_PALETTE = {'No': '#2b5c8f', 'Yes': '#d9534f'}

print("✓ All libraries imported successfully.")
print(f"Pandas {pd.__version__} | NumPy {np.__version__} | Matplotlib {plt.matplotlib.__version__} | Seaborn {sns.__version__}")
"""))

# --- CELL 3: Markdown Section 1 ---
cells.append(new_markdown_cell("""---
## 1. Dataset Ingestion & Systematic Structural Profiling

### **Objective:**
- Load the Telco Customer Churn dataset.
- Inspect dimensions (`shape`), head samples, memory usage, and full statistical descriptions.
- Screen for duplicate rows and null values.
- Conduct a complete unique values audit across all 21 columns.
- Categorize columns into numerical and categorical subsets.
"""))

# --- CELL 4: Code for Loading & Basic Info ---
cells.append(new_code_cell("""# ==========================================================
# LOAD DATASET
# ==========================================================
# Dataset path configuration (handles local environment and standard paths)
DATASET_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
if not os.path.exists(DATASET_PATH):
    # Fallback to kaggle path if executing in Kaggle environment
    DATASET_PATH = "/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATASET_PATH)
print("✓ Dataset Loaded Successfully from:", DATASET_PATH)
print("Dataset Shape (Rows, Columns):", df.shape)
print("\\nFirst 5 Records:")
display(df.head())
"""))

# --- CELL 5: Code for Basic Information & Overview ---
cells.append(new_code_cell("""# ==========================================================
# BASIC INFORMATION & DATASET OVERVIEW
# ==========================================================
print("--- DATASET INFO ---")
df.info()
print("\\n" + "="*80 + "\\n")

print("--- DESCRIPTIVE STATISTICS (ALL ATTRIBUTES) ---")
display(df.describe(include="all").T)
print("\\n" + "="*80 + "\\n")

print("--- DATA TYPES PER COLUMN ---")
print(df.dtypes)
print("\\n" + "="*80 + "\\n")

print("Rows           :", df.shape[0])
print("Columns        :", df.shape[1])
print("Duplicate Rows :", df.duplicated().sum())
print("\\nMissing Values Summary:")
missing = df.isnull().sum().sort_values(ascending=False)
display(missing)
"""))

# --- CELL 6: Code for Unique Values Audit & Column Separation ---
cells.append(new_code_cell("""# ==========================================================
# UNIQUE VALUES AUDIT
# ==========================================================
for col in df.columns:
    print("=" * 55)
    print(f"Column: {col}")
    print(f"Unique Count: {df[col].nunique()}")
    print("Sample Values:", df[col].unique()[:10])

# ==========================================================
# COLUMN TYPES IDENTIFICATION
# ==========================================================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

print("\\n" + "=" * 55)
print("Numerical Features (Pre-cleaning):")
print(numeric_cols)
print("\\nCategorical Features (Pre-cleaning):")
print(categorical_cols)
"""))

# --- CELL 7: Markdown Section 2 ---
cells.append(new_markdown_cell("""---
## 2. Parsing 'TotalCharges', Data Type Rectification & Missing Value Handling

### **Analysis of the Anomaly in `'TotalCharges'`:**
1. In `df.info()`, `'TotalCharges'` is typed as `object` (string) rather than floating point currency.
2. A whitespace check reveals that **11 records** contain blank spaces (`" "`).
3. Cross-referencing with `'tenure'` shows that **every single one of these 11 customers has `tenure == 0`** (newly registered subscribers who have not yet had their first monthly billing cycle).
4. As per standard machine learning preprocessing protocol:
   - Drop the arbitrary identifier column `'customerID'` (zero predictive capacity).
   - Convert whitespace strings in `'TotalCharges'` to `np.nan` and cast to `float64`.
   - Drop the 11 unbilled records (`df.dropna(inplace=True)`), yielding an immaculate dataset of **7,032 rows and 20 columns**.
"""))

# --- CELL 8: Code for TotalCharges Parsing & Cleaning ---
cells.append(new_code_cell("""# ==========================================================
# TOTALCHARGES PARSING & DATA CLEANING
# ==========================================================
# 1. Audit initial data type and whitespace count
print("Initial TotalCharges dtype:", df["TotalCharges"].dtype)
blank_count = (df["TotalCharges"] == " ").sum()
print(f"Count of blank whitespace strings in TotalCharges: {blank_count}")

# 2. Inspect the 11 records with blank TotalCharges
display(df[df["TotalCharges"] == " "][['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'Churn']])

# 3. Drop unique identifier customerID
df.drop(columns=["customerID"], inplace=True)
print("✓ 'customerID' dropped")

# 4. Replace whitespace strings with NaN and cast to float
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

print(f"TotalCharges NaN count prior to drop: {df['TotalCharges'].isnull().sum()}")

# 5. Drop rows containing NaN (the 11 zero-tenure records)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 6. Verify post-cleaning dimensions and null counts
print("\\n--- Post-Cleaning Verification ---")
print("Cleaned DataFrame Shape :", df.shape)
print("Total Remaining Nulls   :", df.isnull().sum().sum())
print("TotalCharges Dtype      :", df["TotalCharges"].dtype)

# 7. Distribution of categories in object columns
print("\\n--- Categorical Feature Distributions ---")
for col in df.select_dtypes(include="object").columns:
    print(f"\\n--- {col} ---")
    print(df[col].value_counts())
"""))

# --- CELL 9: Markdown Section 3 ---
cells.append(new_markdown_cell("""---
## 3. Mandatory Part A Question 1 Analytics: Churn Proportions & Distributions

### **Mandatory Sub-Tasks from Question 1:**
1. **Analyze customer churn proportions across Contract Types** (*Month-to-month, One year, Two year*).
2. **Analyze customer churn proportions across Internet Service Types** (*DSL, Fiber optic, No*).
3. **Generate distribution plots for Tenure vs. Monthly Charges** and bivariate density heatmaps.
4. **Compute correlation matrices** for continuous numerical attributes with collinearity analysis.
"""))

# --- CELL 10: Code for Churn vs Contract Types ---
cells.append(new_code_cell("""# ==========================================================
# 3.1 CHURN PROPORTIONS ACROSS CONTRACT TYPES
# ==========================================================
# Cross-tabulation: Contract vs Churn
contract_ct = pd.crosstab(df['Contract'], df['Churn'], margins=True, margins_name='Total')
contract_pct = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100

contract_summary = contract_ct.copy()
contract_summary['Churn Rate (%)'] = contract_pct['Yes']
print("Contract Type vs Churn Contingency Table:")
display(contract_summary.style.format({'No': '{:,}', 'Yes': '{:,}', 'Total': '{:,}', 'Churn Rate (%)': '{:.2f}%'}))

# Chi-Square Test of Independence
chi2_c, p_c, _, _ = stats.chi2_contingency(pd.crosstab(df['Contract'], df['Churn']))
cramers_v_c = np.sqrt(chi2_c / (len(df) * (min(pd.crosstab(df['Contract'], df['Churn']).shape) - 1)))
print(f"\\nChi-Square Test (Contract vs Churn): Chi2 = {chi2_c:.2f} | p-value = {p_c:.4e} | Cramér's V = {cramers_v_c:.4f}")

# Visualizations: Contract Churn
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Subplot 1: Absolute Customer Count
sns.countplot(data=df, x='Contract', hue='Churn', palette=CHURN_PALETTE, ax=axes[0], order=['Month-to-month', 'One year', 'Two year'])
axes[0].set_title('Customer Count by Contract Type & Churn', fontweight='bold')
axes[0].set_xlabel('Contract Type')
axes[0].set_ylabel('Number of Customers')
axes[0].legend(title='Churn', frameon=True)
for p in axes[0].patches:
    h = p.get_height()
    if h > 0:
        axes[0].annotate(f'{int(h):,}', (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', color='white', fontsize=10, fontweight='bold')

# Subplot 2: 100% Normalized Stacked Bar
contract_pct_sorted = contract_pct.loc[['Month-to-month', 'One year', 'Two year']]
contract_pct_sorted.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=axes[1], edgecolor='black', linewidth=0.5)
axes[1].set_title('100% Normalized Churn Proportion by Contract Type', fontweight='bold')
axes[1].set_xlabel('Contract Type')
axes[1].set_ylabel('Percentage (%)')
axes[1].set_ylim(0, 105)
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='Churn', bbox_to_anchor=(1.02, 1), loc='upper left')

for i, (idx, row) in enumerate(contract_pct_sorted.iterrows()):
    axes[1].text(i, row['No'] / 2, f"{row['No']:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    axes[1].text(i, row['No'] + row['Yes'] / 2, f"{row['Yes']:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()
"""))

# --- CELL 11: Code for Churn vs Internet Service Types ---
cells.append(new_code_cell("""# ==========================================================
# 3.2 CHURN PROPORTIONS ACROSS INTERNET SERVICE TYPES
# ==========================================================
internet_ct = pd.crosstab(df['InternetService'], df['Churn'], margins=True, margins_name='Total')
internet_pct = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100

internet_summary = internet_ct.copy()
internet_summary['Churn Rate (%)'] = internet_pct['Yes']
print("Internet Service Type vs Churn Contingency Table:")
display(internet_summary.style.format({'No': '{:,}', 'Yes': '{:,}', 'Total': '{:,}', 'Churn Rate (%)': '{:.2f}%'}))

# Chi-Square Test
chi2_i, p_i, _, _ = stats.chi2_contingency(pd.crosstab(df['InternetService'], df['Churn']))
cramers_v_i = np.sqrt(chi2_i / (len(df) * (min(pd.crosstab(df['InternetService'], df['Churn']).shape) - 1)))
print(f"\\nChi-Square Test (Internet Service vs Churn): Chi2 = {chi2_i:.2f} | p-value = {p_i:.4e} | Cramér's V = {cramers_v_i:.4f}")

# Visualizations: Internet Service Churn
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

sns.countplot(data=df, x='InternetService', hue='Churn', palette=CHURN_PALETTE, ax=axes[0], order=['Fiber optic', 'DSL', 'No'])
axes[0].set_title('Customer Count by Internet Service & Churn', fontweight='bold')
axes[0].set_xlabel('Internet Service Type')
axes[0].set_ylabel('Number of Customers')
axes[0].legend(title='Churn', frameon=True)
for p in axes[0].patches:
    h = p.get_height()
    if h > 0:
        axes[0].annotate(f'{int(h):,}', (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', color='white', fontsize=10, fontweight='bold')

internet_pct_sorted = internet_pct.loc[['Fiber optic', 'DSL', 'No']]
internet_pct_sorted.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=axes[1], edgecolor='black', linewidth=0.5)
axes[1].set_title('100% Normalized Churn Proportion by Internet Service', fontweight='bold')
axes[1].set_xlabel('Internet Service Type')
axes[1].set_ylabel('Percentage (%)')
axes[1].set_ylim(0, 105)
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='Churn', bbox_to_anchor=(1.02, 1), loc='upper left')

for i, (idx, row) in enumerate(internet_pct_sorted.iterrows()):
    axes[1].text(i, row['No'] / 2, f"{row['No']:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    axes[1].text(i, row['No'] + row['Yes'] / 2, f"{row['Yes']:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()

# Multi-variable interaction: Contract x Internet Service Churn Heatmap
plt.figure(figsize=(9, 4.8))
interaction = df.pivot_table(index='Contract', columns='InternetService', values='Churn', aggfunc=lambda x: (x == 'Yes').mean() * 100)
sns.heatmap(interaction[['DSL', 'Fiber optic', 'No']].loc[['Month-to-month', 'One year', 'Two year']], 
            annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Churn Rate (%)'}, linewidths=1)
plt.title('Churn Rate (%) Heatmap: Contract × Internet Service Interaction', fontweight='bold', pad=12)
plt.xlabel('Internet Service')
plt.ylabel('Contract Type')
plt.show()
"""))

# --- CELL 12: Code for Tenure vs MonthlyCharges Distributions & Correlations ---
cells.append(new_code_cell("""# ==========================================================
# 3.3 TENURE VS MONTHLY CHARGES DISTRIBUTIONS & CORRELATIONS
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bivariate Scatter with Churn Hue
sns.scatterplot(
    data=df, 
    x='tenure', 
    y='MonthlyCharges', 
    hue='Churn', 
    palette=CHURN_PALETTE, 
    alpha=0.6, 
    s=35, 
    ax=axes[0]
)
axes[0].set_title('Tenure vs. Monthly Charges (Stratified by Churn)', fontweight='bold')
axes[0].set_xlabel('Tenure (Months)')
axes[0].set_ylabel('Monthly Charges ($)')
axes[0].axvline(x=12, color='gray', linestyle='--', alpha=0.7, label='1-Year Threshold')
axes[0].axhline(y=70, color='gray', linestyle=':', alpha=0.7, label='High-Charge Threshold ($70)')
axes[0].legend(title='Churn', frameon=True)

# Hexbin Density Plot
hb = axes[1].hexbin(df['tenure'], df['MonthlyCharges'], gridsize=30, cmap='Blues', mincnt=1)
axes[1].set_title('Subscriber Density Concentration (Hexbin)', fontweight='bold')
axes[1].set_xlabel('Tenure (Months)')
axes[1].set_ylabel('Monthly Charges ($)')
fig.colorbar(hb, ax=axes[1], label='Customer Count')

plt.tight_layout()
plt.show()

# Continuous Attribute Correlation Heatmaps (Pearson & Spearman)
cont_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
corr_pearson = df[cont_cols].corr(method='pearson')
corr_spearman = df[cont_cols].corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.heatmap(corr_pearson, annot=True, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0], linewidths=1)
axes[0].set_title('Pearson Correlation Matrix (Linear)', fontweight='bold', pad=12)

sns.heatmap(corr_spearman, annot=True, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1], linewidths=1)
axes[1].set_title('Spearman Rank Correlation Matrix (Monotonic)', fontweight='bold', pad=12)

plt.tight_layout()
plt.show()
"""))

# --- CELL 13: Markdown Section 4 ---
cells.append(new_markdown_cell("""---
## 4. Advanced Feature Engineering (Merged from Preprocessing Pipeline)

### **Engineered Domain Features:**
1. **`TotalServices`**: Aggregates the total number of active services utilized by the customer across Phone, MultipleLines, Internet, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, and StreamingMovies.
2. **`LifetimeValue`**: Economic proxy capturing cumulative revenue potential:
   $$\\text{LifetimeValue} = \\text{MonthlyCharges} \\times \\text{tenure}$$
3. **`AvgMonthlySpend`**: Effective monthly historical spend run-rate:
   $$\\text{AvgMonthlySpend} = \\frac{\\text{TotalCharges}}{\\text{tenure}}$$
4. **`HasInternet`**: Binary indicator designating whether the customer subscribes to an internet infrastructure (`InternetService != "No"`).
5. **`ProtectionScore`**: Bundled cybersecurity and technical care index:
   $$\\text{ProtectionScore} = \\text{OnlineSecurity} + \\text{DeviceProtection} + \\text{TechSupport}$$
"""))

# --- CELL 14: Code for Feature Engineering ---
cells.append(new_code_cell("""# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

# ----------------------------------------------------------
# 1. Total Services
# Count the number of subscribed services across all options
# ----------------------------------------------------------
service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]
df["TotalServices"] = 0
for col in service_columns:
    df["TotalServices"] += df[col].isin(["Yes"]).astype(int)
print("✓ TotalServices created")

# Base services alternative validation (includes base internet connection)
df["TotalServices"] = (
    (df["PhoneService"] == "Yes").astype(int)
    + (df["MultipleLines"] == "Yes").astype(int)
    + (df["InternetService"] != "No").astype(int)
    + (df["OnlineSecurity"] == "Yes").astype(int)
    + (df["OnlineBackup"] == "Yes").astype(int)
    + (df["DeviceProtection"] == "Yes").astype(int)
    + (df["TechSupport"] == "Yes").astype(int)
    + (df["StreamingTV"] == "Yes").astype(int)
    + (df["StreamingMovies"] == "Yes").astype(int)
)

# ----------------------------------------------------------
# 2. Customer Lifetime Value (Proxy)
# ----------------------------------------------------------
df["LifetimeValue"] = (
    df["MonthlyCharges"] *
    df["tenure"]
)
print("✓ LifetimeValue created")

# ----------------------------------------------------------
# 3. Average Monthly Spend
# ----------------------------------------------------------
df["AvgMonthlySpend"] = (
    df["TotalCharges"] /
    df["tenure"]
)
print("✓ AvgMonthlySpend created")

# ----------------------------------------------------------
# 4. Internet User
# ----------------------------------------------------------
df["HasInternet"] = (
    df["InternetService"] != "No"
).astype(int)
print("✓ HasInternet created")

# ----------------------------------------------------------
# 5. Protection Score
# ----------------------------------------------------------
protection_columns = [
    "OnlineSecurity",
    "DeviceProtection",
    "TechSupport"
]
df["ProtectionScore"] = 0
for col in protection_columns:
    df["ProtectionScore"] += (
        df[col] == "Yes"
    ).astype(int)
print("✓ ProtectionScore created")

# Display first 5 rows of engineered features
display(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'LifetimeValue', 'AvgMonthlySpend', 'HasInternet', 'ProtectionScore']].head())
"""))

# --- CELL 15: Markdown Section 5 ---
cells.append(new_markdown_cell("""---
## 5. Binary Encoding, Correlation Analysis & One-Hot Encoding

### **Encoding Strategy:**
1. **Binary Mapping**: Encode binary categorical attributes (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) into $[0, 1]$ numerical flags.
2. **Correlation Heatmap**: Inspect linear dependencies across all numerical and binary features before full dummy expansion.
3. **One-Hot Encoding**: Apply `pd.get_dummies` with `drop_first=True, dtype=int` on multi-class categorical predictors (`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`).
4. **Dataset Export**: Save the resulting $(7032, 36)$ clean matrix to `telco_preprocessed.csv`.
"""))

# --- CELL 16: Code for Binary Encoding, Heatmap & One-Hot Encoding ---
cells.append(new_code_cell("""# ==========================================================
# LABEL / BINARY ENCODING
# ==========================================================
binary_mapping = {
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0
}

binary_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]

for col in binary_columns:
    df[col] = df[col].map(binary_mapping)
print("✓ Binary columns encoded")

# ==========================================================
# CORRELATION HEATMAP (NUMERIC & BINARY ATTRIBUTES)
# ==========================================================
plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap: Numerical & Binary Engineered Features", fontweight='bold', pad=12)
plt.show()

# ==========================================================
# ONE-HOT ENCODING
# ==========================================================
multi_columns = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

df = pd.get_dummies(
    df,
    columns=multi_columns,
    drop_first=True,
    dtype=int
)
print("✓ One-Hot Encoding completed")

print("\\nFinal Preprocessed Dataset Dimensions (Rows, Columns):", df.shape)
print("\\nFirst 5 Records of Preprocessed Dataset:")
display(df.head())

print("\\nFinal Schema Information:")
df.info()

print("\\nTarget Variable ('Churn') Distribution:")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)

# ==========================================================
# SAVE PREPROCESSED DATASET
# ==========================================================
df.to_csv("telco_preprocessed.csv", index=False)
print("✓ Preprocessed dataset successfully saved to 'telco_preprocessed.csv'!")
"""))

# --- CELL 17: Markdown Section 6 ---
cells.append(new_markdown_cell("""---
## 6. Interactive Interface for Data Inspection & Parameter Tuning

### **Evaluation Rubric Compliance:**
> ***"provide an interface to load dataset, training, parameter adjustment and for visualization."***

The widget control interface below enables interactive cohort exploration:
- **Contract Type & Internet Service Selectors**: Filter subscribers dynamically.
- **Tenure & Monthly Charges Range Sliders**: Adjust observation bounds in real time.
- **Real-Time KPI Scorecards**: Immediate calculation of filtered cohort count, churn rate, mean tenure, and mean spend.
- **Dynamic 2-Panel Plots**: Real-time rendering of bivariate scatter plots and cohort churn bars.
"""))

# --- CELL 18: Code for Interactive GUI Widget ---
cells.append(new_code_cell("""# Interactive GUI Control Dashboard using ipywidgets
# Load unencoded copy for clean widget category names
df_ui = pd.read_csv(DATASET_PATH)
df_ui['TotalCharges'] = pd.to_numeric(df_ui['TotalCharges'].replace(" ", np.nan))
df_ui.dropna(subset=['TotalCharges'], inplace=True)

contract_opts = ['All'] + list(df_ui['Contract'].unique())
internet_opts = ['All'] + list(df_ui['InternetService'].unique())

w_contract = widgets.Dropdown(options=contract_opts, value='All', description='Contract:', style={'description_width': '90px'})
w_internet = widgets.Dropdown(options=internet_opts, value='All', description='Internet:', style={'description_width': '90px'})
w_tenure = widgets.IntRangeSlider(
    value=[0, int(df_ui['tenure'].max())],
    min=0, max=int(df_ui['tenure'].max()), step=1,
    description='Tenure (Mo):', style={'description_width': '90px'}, layout=widgets.Layout(width='380px')
)
w_charges = widgets.FloatRangeSlider(
    value=[float(df_ui['MonthlyCharges'].min()), float(df_ui['MonthlyCharges'].max())],
    min=float(df_ui['MonthlyCharges'].min()), max=float(df_ui['MonthlyCharges'].max()), step=1.0,
    description='Charges ($):', style={'description_width': '90px'}, layout=widgets.Layout(width='380px')
)

ui_out = widgets.Output()

def update_ui(change=None):
    with ui_out:
        clear_output(wait=True)
        sub = df_ui.copy()
        if w_contract.value != 'All':
            sub = sub[sub['Contract'] == w_contract.value]
        if w_internet.value != 'All':
            sub = sub[sub['InternetService'] == w_internet.value]
        sub = sub[(sub['tenure'] >= w_tenure.value[0]) & (sub['tenure'] <= w_tenure.value[1]) &
                  (sub['MonthlyCharges'] >= w_charges.value[0]) & (sub['MonthlyCharges'] <= w_charges.value[1])]
        
        n = len(sub)
        if n == 0:
            print("No records match the active filters.")
            return
            
        c_count = (sub['Churn'] == 'Yes').sum()
        c_rate = (c_count / n) * 100
        
        banner = f\"\"\"
        <div style="display: flex; gap: 15px; margin-bottom: 12px;">
            <div style="background-color: #f0f4f8; padding: 10px 18px; border-radius: 8px; border-left: 5px solid #2b5c8f;">
                <span style="font-size: 11px; color: #555;">COHORT SIZE</span><br>
                <b style="font-size: 18px; color: #1a365d;">{n:,}</b>
            </div>
            <div style="background-color: #fdf0f0; padding: 10px 18px; border-radius: 8px; border-left: 5px solid #d9534f;">
                <span style="font-size: 11px; color: #555;">CHURN RATE</span><br>
                <b style="font-size: 18px; color: #c53030;">{c_rate:.2f}%</b> ({c_count:,})
            </div>
            <div style="background-color: #f0fdf4; padding: 10px 18px; border-radius: 8px; border-left: 5px solid #38a169;">
                <span style="font-size: 11px; color: #555;">MEAN TENURE</span><br>
                <b style="font-size: 18px; color: #22543d;">{sub['tenure'].mean():.1f} mo</b>
            </div>
            <div style="background-color: #fefcbf; padding: 10px 18px; border-radius: 8px; border-left: 5px solid #d69e2e;">
                <span style="font-size: 11px; color: #555;">MEAN CHARGE</span><br>
                <b style="font-size: 18px; color: #744210;">${sub['MonthlyCharges'].mean():.2f}</b>
            </div>
        </div>
        \"\"\"
        display(HTML(banner))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
        sns.scatterplot(data=sub, x='tenure', y='MonthlyCharges', hue='Churn', palette=CHURN_PALETTE, alpha=0.7, s=35, ax=axes[0])
        axes[0].set_title(f'Filtered Cohort Scatter Plot (N={n:,})', fontweight='bold')
        axes[0].set_xlabel('Tenure (Months)')
        axes[0].set_ylabel('Monthly Charges ($)')
        
        p = sub['Churn'].value_counts(normalize=True) * 100
        bars = axes[1].bar(['No Churn', 'Churned'], [p.get('No', 0), p.get('Yes', 0)], color=['#2b5c8f', '#d9534f'], width=0.45)
        axes[1].set_title('Cohort Churn Rate (%)', fontweight='bold')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_ylim(0, 105)
        for b in bars:
            h = b.get_height()
            axes[1].text(b.get_x() + b.get_width()/2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        plt.show()

w_contract.observe(update_ui, names='value')
w_internet.observe(update_ui, names='value')
w_tenure.observe(update_ui, names='value')
w_charges.observe(update_ui, names='value')

controls = widgets.VBox([
    widgets.HTML("<b>Interactive Parameter Adjustment & Exploration Interface:</b>"),
    widgets.HBox([w_contract, w_internet]),
    widgets.HBox([w_tenure, w_charges])
])

display(controls)
display(ui_out)
update_ui()
"""))

# --- CELL 19: Markdown Section 7 ---
cells.append(new_markdown_cell("""---
## 7. Synthesis, Key Findings & Roadmap for Part B

### **Key Findings from Part A:**
1. **TotalCharges Whitespace Anomaly & Data Cleansing**:
   - Exactly 11 records contained `" "` blank strings in `TotalCharges`.
   - All 11 records had `tenure == 0` (unbilled accounts).
   - Following standard data science preprocessing protocol, dropping these unbilled accounts and discarding `'customerID'` produced a clean matrix of $7,032$ observations.
2. **Contract Type as the Leading Churn Predictor**:
   - Month-to-month contracts exhibit an alarming **42.71%** churn rate, compared to **11.27%** for 1-year and **2.83%** for 2-year contracts.
   - $\chi^2$ Test of Independence ($p < 10^{-200}$) confirms massive statistical significance.
3. **Fiber Optic Service Attrition Risk**:
   - Fiber optic subscribers exhibit a **41.89%** churn rate, far higher than DSL ($18.96%$) and No Internet ($7.40%$).
   - In combination, **Month-to-month + Fiber Optic subscribers exhibit a peak churn rate of 54.6%**.
4. **Engineered Feature Value**:
   - Engineered domain features (`TotalServices`, `LifetimeValue`, `AvgMonthlySpend`, `HasInternet`, `ProtectionScore`) capture multi-service stickiness and customer account maturity.
   - One-hot encoding expanded multi-class categorical predictors into 36 numeric modeling columns.

---

### **Strategic Roadmap for Part B (Model Fitting & Cross-Validation):**
- **Class Imbalance Management**: The target variable has an approximate 73:27 imbalance ($5,163$ No vs $1,869$ Yes). We will apply **SMOTE** and class-weight balancing.
- **Model Training**: Construct **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Random Forest** classifiers.
- **Validation**: Perform **5-Fold Stratified Cross-Validation** with hyperparameter tuning.
"""))

nb.cells = cells

# Save merged notebook
target_nb_path = "26MML0020_Rohan_ML_MiniProject_PartA.ipynb"
with open(target_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Merged notebook created at {target_nb_path}. Now executing cells to populate output...")

# Execute notebook so all cell outputs are pre-rendered
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

with open(target_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully executed and saved: {target_nb_path}")
