# ============================================================
# DAY 2 — FIX: Regenerate visualizations only
# Run this after day2_model_training.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import shap

plt.style.use('seaborn-v0_8-whitegrid')
BLUE = '#2563eb'
RED  = '#dc2626'
AMB  = '#f59e0b'

# ── Load saved models and data ───────────────────────────────
print("Loading models and data...")
rf       = joblib.load('rf_model.pkl')
xgb      = joblib.load('xgb_model.pkl')
scaler   = joblib.load('scaler.pkl')
FEATURES = joblib.load('features.pkl')

df = pd.read_csv('cs-training-clean.csv')
df = df.drop(columns=['AgeGroup'], errors='ignore')

TARGET = 'SeriousDlqin2yrs'
X = df[FEATURES]
y = df[TARGET]

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

X_train_scaled = scaler.transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train_sm)

lr_proba  = lr.predict_proba(X_test_scaled)[:, 1]
rf_proba  = rf.predict_proba(X_test)[:, 1]
xgb_proba = xgb.predict_proba(X_test)[:, 1]
rf_pred   = rf.predict(X_test)

lr_auc  = roc_auc_score(y_test, lr_proba)
rf_auc  = roc_auc_score(y_test, rf_proba)
xgb_auc = roc_auc_score(y_test, xgb_proba)

results = {
    'Logistic Regression': lr_auc,
    'Random Forest':       rf_auc,
    'XGBoost':             xgb_auc,
}

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

# ── SHAP (fixed) ─────────────────────────────────────────────
print("Computing SHAP values...")
explainer  = shap.TreeExplainer(rf)
X_sample   = X_test.sample(500, random_state=42)
shap_values = explainer.shap_values(X_sample)

# Fix: handle (n_samples, n_features, 2) or list format
if isinstance(shap_values, list):
    sv = shap_values[1]                        # list format → take class 1
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]                  # 3D array → take class 1 slice
else:
    sv = shap_values                            # already 2D

print(f"  SHAP values shape: {sv.shape}")

# ── Figure 1: Main results (6 charts) ────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Credit Risk AI — Model Training Results', fontsize=16, fontweight='bold')

# Plot 1: AUC-ROC bar
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
    ('Logistic Reg.', lr_proba,  'gray', '--'),
    ('Random Forest', rf_proba,  BLUE,   '-'),
    ('XGBoost',       xgb_proba, RED,    '-'),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
            label=f'{name} (AUC={roc_auc_score(y_test, proba):.3f})')
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4)
ax.set_title('ROC Curves — All Models', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9)

# Plot 3: Confusion matrix
ax = axes[0, 2]
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
       display_labels=['No Default', 'Defaulted'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Random Forest', fontweight='bold')

# Plot 4: Feature importance
ax = axes[1, 0]
top10 = importances.head(10)
colors_fi = [RED if i == 0 else BLUE for i in range(len(top10))]
ax.barh(top10.index[::-1], top10.values[::-1], color=colors_fi[::-1], edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontweight='bold')
ax.set_xlabel('Importance Score')

# Plot 5: SMOTE before/after
ax = axes[1, 1]
categories = ['Before SMOTE', 'After SMOTE']
no_def  = [y_train.value_counts()[0], y_train_sm.value_counts()[0]]
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

# Plot 6: SHAP mean impact (fixed)
ax = axes[1, 2]
mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False).head(10)
ax.barh(mean_shap.index[::-1], mean_shap.values[::-1], color=AMB, edgecolor='white')
ax.set_title('SHAP — Mean Feature Impact', fontweight='bold')
ax.set_xlabel('Mean |SHAP value|')

plt.tight_layout()
plt.savefig('day2_model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: day2_model_results.png")

# ── Figure 2: SHAP bar summary ────────────────────────────────
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_sample, plot_type='bar',
                  feature_names=FEATURES, show=False)
plt.title('SHAP Feature Importance (Bar)', fontweight='bold')
plt.tight_layout()
plt.savefig('day2_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: day2_shap_bar.png")

# ── Final summary ─────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════╗
  DAY 2 COMPLETE
  Best Model  : Random Forest
  AUC-ROC     : {rf_auc:.4f}
  Charts saved: day2_model_results.png
                day2_shap_bar.png
  Ready for Day 3 → Streamlit App
╚══════════════════════════════════════════════╝
""")