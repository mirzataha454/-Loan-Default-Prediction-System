# ============================================================
# DAY 2 — Credit Risk Assessment
# Model Training: Random Forest + XGBoost + SHAP
# Input: cs-training-clean.csv (from Day 1)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import joblib

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# STEP 1 — LOAD CLEAN DATA
# ============================================================
print("=" * 55)
print("STEP 1: Loading clean dataset")
print("=" * 55)

df = pd.read_csv('cs-training-clean.csv')
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Drop non-numeric columns for modeling
df = df.drop(columns=['AgeGroup'], errors='ignore')

# ============================================================
# STEP 2 — PREPARE FEATURES & TARGET
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: Preparing features and target")
print("=" * 55)

TARGET = 'SeriousDlqin2yrs'
FEATURES = [col for col in df.columns if col != TARGET]

X = df[FEATURES]
y = df[TARGET]

print(f"  Features used ({len(FEATURES)}):")
for f in FEATURES:
    print(f"    • {f}")
print(f"\n  Target: {TARGET}")
print(f"  Class distribution: {dict(y.value_counts())}")

# ============================================================
# STEP 3 — TRAIN / TEST SPLIT
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: Train / Test split (80/20)")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set : {X_train.shape[0]:,} rows")
print(f"  Test set     : {X_test.shape[0]:,} rows")
print(f"  Defaults in train: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")

# ============================================================
# STEP 4 — SMOTE (fix class imbalance)
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: Applying SMOTE to balance classes")
print("=" * 55)

print(f"  Before SMOTE — Class 0: {(y_train==0).sum():,}  Class 1: {(y_train==1).sum():,}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"  After  SMOTE — Class 0: {(y_train_sm==0).sum():,}  Class 1: {(y_train_sm==1).sum():,}")
print(f"  New training size: {X_train_sm.shape[0]:,} rows")
print("  ✓ Classes are now balanced 50/50")

# ============================================================
# STEP 5 — TRAIN 3 MODELS
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: Training models")
print("=" * 55)

# -- Logistic Regression (baseline) --------------------------
print("\n  [1/3] Logistic Regression (baseline)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train_sm)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
print(f"      AUC-ROC: {lr_auc:.4f}")

# -- Random Forest (primary) ---------------------------------
print("\n  [2/3] Random Forest (primary model)...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sm, y_train_sm)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred  = rf.predict(X_test)
rf_auc   = roc_auc_score(y_test, rf_proba)
print(f"      AUC-ROC: {rf_auc:.4f}")

# -- XGBoost (challenger) ------------------------------------
print("\n  [3/3] XGBoost (challenger)...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_sm, y_train_sm,
        eval_set=[(X_test, y_test)],
        verbose=False)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_pred  = xgb.predict(X_test)
xgb_auc   = roc_auc_score(y_test, xgb_proba)
print(f"      AUC-ROC: {xgb_auc:.4f}")

# ============================================================
# STEP 6 — MODEL COMPARISON
# ============================================================
print("\n" + "=" * 55)
print("STEP 6: Model Comparison")
print("=" * 55)

results = {
    'Logistic Regression': lr_auc,
    'Random Forest':       rf_auc,
    'XGBoost':             xgb_auc,
}

print(f"\n  {'Model':<25} {'AUC-ROC':>10}  {'Rating':>10}")
print("  " + "-" * 50)
for model, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    stars = '★' * int(auc * 5)
    winner = ' ← WINNER' if auc == max(results.values()) else ''
    print(f"  {model:<25} {auc:>10.4f}  {stars}{winner}")

best_model   = rf if rf_auc >= xgb_auc else xgb
best_name    = 'Random Forest' if rf_auc >= xgb_auc else 'XGBoost'
best_proba   = rf_proba if rf_auc >= xgb_auc else xgb_proba
best_pred    = rf_pred if rf_auc >= xgb_auc else xgb_pred

print(f"\n  Best model: {best_name} (AUC = {max(rf_auc, xgb_auc):.4f})")

# ============================================================
# STEP 7 — CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 55)
print(f"STEP 7: Classification Report — {best_name}")
print("=" * 55)
print(classification_report(y_test, best_pred,
      target_names=['No Default', 'Defaulted']))

# ============================================================
# STEP 8 — FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 55)
print("STEP 8: Feature Importance (Random Forest)")
print("=" * 55)

importances = pd.Series(rf.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

print("\n  Feature importances:")
for feat, imp in importances.items():
    bar = '█' * int(imp * 200)
    print(f"  {feat:<42} {imp:.4f}  {bar}")

# ============================================================
# STEP 9 — SHAP EXPLAINABILITY
# ============================================================
print("\n" + "=" * 55)
print("STEP 9: SHAP Explainability")
print("=" * 55)

print("  Computing SHAP values (this takes ~30 seconds)...")
explainer  = shap.TreeExplainer(rf)
X_sample   = X_test.sample(500, random_state=42)
shap_values = explainer.shap_values(X_sample)

# Handle both old and new shap output formats
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

print("  ✓ SHAP values computed for 500 test samples")

# ============================================================
# STEP 10 — SAVE MODEL + ARTIFACTS
# ============================================================
print("\n" + "=" * 55)
print("STEP 10: Saving model and artifacts")
print("=" * 55)

joblib.dump(rf,      'rf_model.pkl')
joblib.dump(xgb,     'xgb_model.pkl')
joblib.dump(scaler,  'scaler.pkl')
joblib.dump(FEATURES,'features.pkl')

print("  ✓ rf_model.pkl")
print("  ✓ xgb_model.pkl")
print("  ✓ scaler.pkl")
print("  ✓ features.pkl")

# ============================================================
# STEP 11 — VISUALIZATIONS (for PPT)
# ============================================================
print("\n" + "=" * 55)
print("STEP 11: Generating visualizations")
print("=" * 55)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Credit Risk AI — Model Training Results', fontsize=16, fontweight='bold')

BLUE = '#2563eb'
RED  = '#dc2626'
AMB  = '#f59e0b'

# Plot 1: AUC-ROC comparison bar
ax = axes[0, 0]
models = list(results.keys())
aucs   = list(results.values())
colors = [RED if v == max(aucs) else BLUE for v in aucs]
bars = ax.bar(models, aucs, color=colors, edgecolor='white', linewidth=0.5)
ax.set_ylim(0.5, 1.0)
ax.set_title('AUC-ROC Comparison', fontweight='bold')
ax.set_ylabel('AUC-ROC Score')
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{auc:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=10)

# Plot 2: ROC curves
ax = axes[0, 1]
for name, proba, color, ls in [
    ('Logistic Reg.', lr_proba, 'gray', '--'),
    ('Random Forest', rf_proba, BLUE,  '-'),
    ('XGBoost',       xgb_proba, RED,  '-'),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = results[name] if name != 'Logistic Reg.' else lr_auc
    ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
            label=f'{name} (AUC={roc_auc_score(y_test, proba):.3f})')
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4)
ax.set_title('ROC Curves — All Models', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9)

# Plot 3: Confusion matrix
ax = axes[0, 2]
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
       display_labels=['No Default', 'Defaulted'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold')

# Plot 4: Feature importance (top 10)
ax = axes[1, 0]
top10 = importances.head(10)
colors_fi = [RED if i == 0 else BLUE for i in range(len(top10))]
bars = ax.barh(top10.index[::-1], top10.values[::-1], color=colors_fi[::-1], edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontweight='bold')
ax.set_xlabel('Importance Score')

# Plot 5: SMOTE before/after
ax = axes[1, 1]
categories = ['Before SMOTE\n(Training)', 'After SMOTE\n(Training)']
no_def = [y_train.value_counts()[0], y_train_sm.value_counts()[0]]
yes_def = [y_train.value_counts()[1], y_train_sm.value_counts()[1]]
x = np.arange(len(categories))
w = 0.35
ax.bar(x - w/2, no_def,  w, label='No Default', color=BLUE, edgecolor='white')
ax.bar(x + w/2, yes_def, w, label='Defaulted',  color=RED,  edgecolor='white')
ax.set_title('SMOTE — Class Balancing Effect', fontweight='bold')
ax.set_ylabel('Sample Count')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
for i, (nd, yd) in enumerate(zip(no_def, yes_def)):
    ax.text(i - w/2, nd + 300, f'{nd:,}', ha='center', fontsize=9)
    ax.text(i + w/2, yd + 300, f'{yd:,}', ha='center', fontsize=9)

# Plot 6: SHAP summary (mean absolute)
ax = axes[1, 2]
mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False).head(10)
ax.barh(mean_shap.index[::-1], mean_shap.values[::-1], color=AMB, edgecolor='white')
ax.set_title('SHAP — Mean Feature Impact', fontweight='bold')
ax.set_xlabel('Mean |SHAP value|')

plt.tight_layout()
plt.savefig('day2_model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: day2_model_results.png")

# SHAP waterfall for one sample
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_sample, plot_type='bar',
                  feature_names=FEATURES, show=False)
plt.title('SHAP Feature Importance (Bar)', fontweight='bold')
plt.tight_layout()
plt.savefig('day2_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: day2_shap_bar.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("DAY 2 COMPLETE — Summary")
print("=" * 55)
print(f"""
  Models trained    : Logistic Regression, Random Forest, XGBoost
  SMOTE applied     : {y_train.sum():,} → {y_train_sm.sum():,} default samples
  Best model        : {best_name}
  Best AUC-ROC      : {max(rf_auc, xgb_auc):.4f}
  SHAP values       : Computed for 500 test samples

  Saved files:
    rf_model.pkl          ← main model for the app
    xgb_model.pkl         ← challenger model
    scaler.pkl            ← for logistic regression
    features.pkl          ← feature list for app
    day2_model_results.png
    day2_shap_bar.png

  Ready for Day 3 → Streamlit Web App
""")