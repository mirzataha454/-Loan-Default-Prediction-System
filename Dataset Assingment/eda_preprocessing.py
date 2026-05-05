# ============================================================
# DAY 1 — Credit Risk Assessment
# EDA + Preprocessing
# Dataset: cs-training.csv (Give Me Some Credit — Kaggle)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2563eb', '#dc2626']  # blue = no default, red = default

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
print("=" * 55)
print("STEP 1: Loading dataset")
print("=" * 55)

df = pd.read_csv('cs-training.csv', index_col=0)

print(f"Shape          : {df.shape}")
print(f"Rows           : {df.shape[0]:,}")
print(f"Columns        : {df.shape[1]}")
print("\nColumn names:")
for col in df.columns:
    print(f"  • {col}")

# ============================================================
# STEP 2 — BASIC OVERVIEW
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: Basic Overview")
print("=" * 55)

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Statistical Summary ---")
print(df.describe().round(2).to_string())

# ============================================================
# STEP 3 — TARGET VARIABLE (class imbalance check)
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: Target Variable — SeriousDlqin2yrs")
print("=" * 55)

target_counts = df['SeriousDlqin2yrs'].value_counts()
target_pct    = df['SeriousDlqin2yrs'].value_counts(normalize=True) * 100

print(f"\n  No Default (0) : {target_counts[0]:,}  ({target_pct[0]:.1f}%)")
print(f"  Defaulted  (1) : {target_counts[1]:,}  ({target_pct[1]:.1f}%)")
print(f"\n  ⚠ Class imbalance ratio: {target_counts[0]/target_counts[1]:.1f}:1")
print("  → We will use SMOTE on Day 2 to fix this.")

# ============================================================
# STEP 4 — MISSING VALUES
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: Missing Values")
print("=" * 55)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).query('`Missing Count` > 0')

if missing_df.empty:
    print("  No missing values found.")
else:
    print(missing_df.to_string())
    print("\n  Strategy:")
    print("  • MonthlyIncome  → fill with MEDIAN (skewed distribution)")
    print("  • NumberOfDependents → fill with MODE (categorical-like)")

# ── Fix missing values ───────────────────────────────────────
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0], inplace=True)

print(f"\n  ✓ Missing values after fix: {df.isnull().sum().sum()}")

# ============================================================
# STEP 5 — OUTLIER DETECTION
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: Outlier Detection")
print("=" * 55)

outlier_cols = [
    'RevolvingUtilizationOfUnsecuredLines',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
]

for col in outlier_cols:
    q99 = df[col].quantile(0.99)
    outliers = (df[col] > q99).sum()
    print(f"  {col[:45]:<45} | 99th pct: {q99:>10.2f} | Outliers: {outliers:>5,}")

# Cap extreme values at 99th percentile
print("\n  Capping outliers at 99th percentile...")
for col in outlier_cols:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)
print("  ✓ Outliers capped.")

# ============================================================
# STEP 6 — FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 55)
print("STEP 6: Feature Engineering")
print("=" * 55)

# New feature 1: Total late payments
df['TotalLatePayments'] = (
    df['NumberOfTime30-59DaysPastDueNotWorse'] +
    df['NumberOfTime60-89DaysPastDueNotWorse'] +
    df['NumberOfTimes90DaysLate']
)

# New feature 2: Income per dependent (avoids division by zero)
df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)

# New feature 3: Age group buckets
df['AgeGroup'] = pd.cut(
    df['age'],
    bins=[0, 30, 45, 60, 120],
    labels=['Young', 'Middle', 'Senior', 'Elder']
)

# New feature 4: High utilization flag
df['HighUtilization'] = (df['RevolvingUtilizationOfUnsecuredLines'] > 0.75).astype(int)

print("  New features added:")
print("  ✓ TotalLatePayments    — sum of all delinquency columns")
print("  ✓ IncomePerDependent   — monthly income / (dependents + 1)")
print("  ✓ AgeGroup             — categorical age buckets")
print("  ✓ HighUtilization      — flag if utilization > 75%")

# ============================================================
# STEP 7 — CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 55)
print("STEP 7: Correlation with Target")
print("=" * 55)

numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()['SeriousDlqin2yrs'].drop('SeriousDlqin2yrs').sort_values(ascending=False)

print("\n  Feature correlations with default (SeriousDlqin2yrs):")
for feat, corr in correlations.items():
    bar = '█' * int(abs(corr) * 40)
    sign = '+' if corr > 0 else '-'
    print(f"  {feat[:42]:<42} {sign}{abs(corr):.4f}  {bar}")

# ============================================================
# STEP 8 — SAVE CLEAN DATASET
# ============================================================
print("\n" + "=" * 55)
print("STEP 8: Saving clean dataset")
print("=" * 55)

df.to_csv('cs-training-clean.csv', index=False)
print(f"  ✓ Saved: cs-training-clean.csv")
print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================
# STEP 9 — VISUALIZATIONS (saves as PNG files)
# ============================================================
print("\n" + "=" * 55)
print("STEP 9: Generating visualizations")
print("=" * 55)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Credit Risk Dataset — EDA Overview', fontsize=16, fontweight='bold', y=1.01)

# Plot 1: Class distribution
ax = axes[0, 0]
bars = ax.bar(['No Default (0)', 'Defaulted (1)'], target_counts.values, color=COLORS, edgecolor='white', linewidth=0.5)
ax.set_title('Target Variable Distribution', fontweight='bold')
ax.set_ylabel('Count')
for bar, pct in zip(bars, target_pct.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Plot 2: Age distribution by default
ax = axes[0, 1]
df[df['SeriousDlqin2yrs'] == 0]['age'].plot.hist(ax=ax, bins=30, alpha=0.6, color=COLORS[0], label='No Default')
df[df['SeriousDlqin2yrs'] == 1]['age'].plot.hist(ax=ax, bins=30, alpha=0.6, color=COLORS[1], label='Defaulted')
ax.set_title('Age Distribution by Default Status', fontweight='bold')
ax.set_xlabel('Age')
ax.legend()

# Plot 3: Revolving utilization
ax = axes[0, 2]
df[df['SeriousDlqin2yrs'] == 0]['RevolvingUtilizationOfUnsecuredLines'].plot.hist(
    ax=ax, bins=30, alpha=0.6, color=COLORS[0], label='No Default')
df[df['SeriousDlqin2yrs'] == 1]['RevolvingUtilizationOfUnsecuredLines'].plot.hist(
    ax=ax, bins=30, alpha=0.6, color=COLORS[1], label='Defaulted')
ax.set_title('Revolving Utilization by Default Status', fontweight='bold')
ax.set_xlabel('Utilization Rate')
ax.legend()

# Plot 4: Correlation heatmap
ax = axes[1, 0]
num_cols = df.select_dtypes(include=[np.number]).columns[:10]
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, ax=ax, cmap='RdBu_r', center=0,
            annot=False, linewidths=0.3, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0, labelsize=7)

# Plot 5: Total late payments vs default
ax = axes[1, 1]
late_default = df.groupby('TotalLatePayments')['SeriousDlqin2yrs'].mean().head(10)
ax.bar(late_default.index, late_default.values * 100, color='#f59e0b', edgecolor='white')
ax.set_title('Default Rate by Total Late Payments', fontweight='bold')
ax.set_xlabel('Number of Late Payments')
ax.set_ylabel('Default Rate (%)')

# Plot 6: Monthly income distribution (log scale)
ax = axes[1, 2]
income_no_default = df[df['SeriousDlqin2yrs'] == 0]['MonthlyIncome']
income_default    = df[df['SeriousDlqin2yrs'] == 1]['MonthlyIncome']
ax.hist(np.log1p(income_no_default), bins=30, alpha=0.6, color=COLORS[0], label='No Default')
ax.hist(np.log1p(income_default),    bins=30, alpha=0.6, color=COLORS[1], label='Defaulted')
ax.set_title('Monthly Income (log scale) by Default', fontweight='bold')
ax.set_xlabel('log(Monthly Income + 1)')
ax.legend()

plt.tight_layout()
plt.savefig('day1_eda_charts.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: day1_eda_charts.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("DAY 1 COMPLETE — Summary")
print("=" * 55)
print(f"""
  Dataset loaded    : 150,000 rows, 11 original features
  Missing values    : Fixed (median/mode imputation)
  Outliers          : Capped at 99th percentile
  New features      : 4 engineered features added
  Class imbalance   : ~13:1 (will fix with SMOTE on Day 2)
  Clean file saved  : cs-training-clean.csv
  Charts saved      : day1_eda_charts.png

  Ready for Day 2 → Model Training (Random Forest + XGBoost)
""")