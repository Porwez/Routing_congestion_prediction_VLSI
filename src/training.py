
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "realistic_congestion_dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 300

print("=" * 70)
print("VLSI ROUTING CONGESTION - TRAINING PIPELINE")
print("=" * 70)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\n[1/8] Loading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Run the dataset generation script first."
    )

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "layout_id",
    "tech_node",
    "congestion",
    "density",
    "pin_density",
    "fanout",
    "macro",
    "capacity",
    "macro_proximity",
    "wire_length",
    "clock_region",
    "drc_violations",
    "net_criticality",
    "power_density",
    "temperature",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

df = df.dropna(subset=required_columns).copy()

print(f"Dataset shape: {df.shape}")
print(f"Samples: {len(df):,}")
print(f"Layouts: {df['layout_id'].nunique()}")
print(f"Technology nodes: {df['tech_node'].unique()}")

print("\nCongestion statistics:")
print(df["congestion"].describe())

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[2/8] Generating dataset analysis...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Congestion distribution
axes[0].hist(
    df["congestion"],
    bins=50,
    edgecolor="black",
    alpha=0.75,
)
axes[0].set_title("Congestion Distribution")
axes[0].set_xlabel("Congestion")
axes[0].set_ylabel("Frequency")
axes[0].grid(alpha=0.25)

# Congestion by technology node
for tech in sorted(df["tech_node"].unique()):
    values = df.loc[df["tech_node"] == tech, "congestion"]
    axes[1].hist(values, bins=30, alpha=0.45, label=tech)

axes[1].set_title("Congestion by Technology Node")
axes[1].set_xlabel("Congestion")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].grid(alpha=0.25)

# Boxplot using current Matplotlib API
tech_nodes = sorted(df["tech_node"].unique())
tech_data = [
    df.loc[df["tech_node"] == tech, "congestion"].values
    for tech in tech_nodes
]

axes[2].boxplot(tech_data, tick_labels=tech_nodes)
axes[2].set_xlabel("Technology Node")
axes[2].set_ylabel("Congestion")
axes[2].set_title("Congestion Range by Technology")
axes[2].grid(alpha=0.25, axis="y")

plt.tight_layout()
plt.savefig("congestion_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved congestion_distribution.png")

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================

print("\n[3/8] Engineering features...")

# LabelEncoder preserves the original project's encoding approach.
label_encoder = LabelEncoder()
df["tech_node_encoded"] = label_encoder.fit_transform(
    df["tech_node"].astype(str)
)

def classify_congestion(value):
    if value < 0.8:
        return "Green"
    elif value < 1.5:
        return "Yellow"
    return "Red"

df["congestion_class"] = df["congestion"].apply(classify_congestion)

print("\nTechnology node encoding:")
print({
    node: int(encoded)
    for node, encoded in zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_),
    )
})

print("\nCongestion class distribution:")
print(df["congestion_class"].value_counts())

feature_cols = [
    "density",
    "pin_density",
    "fanout",
    "macro",
    "capacity",
    "macro_proximity",
    "wire_length",
    "clock_region",
    "drc_violations",
    "net_criticality",
    "power_density",
    "temperature",
    "tech_node_encoded",
]

X = df[feature_cols].astype(float)
y_reg = df["congestion"].astype(float)
y_cls = df["congestion_class"]

# ============================================================
# STEP 4: TRAIN / TEST SPLIT
# ============================================================

print("\n[4/8] Splitting dataset...")

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["tech_node"],
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train_reg = y_reg.iloc[train_idx]
y_test_reg = y_reg.iloc[test_idx]

y_train_cls = y_cls.iloc[train_idx]
y_test_cls = y_cls.iloc[test_idx]

print(f"Training samples: {len(train_idx):,}")
print(f"Testing samples:  {len(test_idx):,}")
print(f"Features:         {len(feature_cols)}")

# ============================================================
# STEP 5: TRAIN REGRESSION MODEL
# ============================================================

print("\n[5/8] Training Random Forest Regressor...")

rf_reg = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

rf_reg.fit(X_train, y_train_reg)
y_pred_reg = rf_reg.predict(X_test)

mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

# MAPE can be unstable when actual values are near zero.
mape = (
    np.mean(
        np.abs(
            (y_test_reg.to_numpy() - y_pred_reg)
            / np.maximum(np.abs(y_test_reg.to_numpy()), 1e-6)
        )
    )
    * 100
)

print("\nREGRESSION PERFORMANCE")
print("-" * 40)
print(f"MSE:      {mse:.6f}")
print(f"RMSE:     {rmse:.6f}")
print(f"MAE:      {mae:.6f}")
print(f"MAPE:     {mape:.3f}%")
print(f"R2 score: {r2:.6f}")

print("\nRunning 5-fold cross-validation...")
cv_scores = cross_val_score(
    rf_reg,
    X_train,
    y_train_reg,
    cv=5,
    scoring="r2",
    n_jobs=-1,
)

print("CV R2 scores:", cv_scores)
print(f"Mean CV R2: {cv_scores.mean():.6f}")
print(f"CV standard deviation: {cv_scores.std():.6f}")

# ============================================================
# STEP 6: TRAIN CLASSIFICATION MODEL
# ============================================================

print("\n[6/8] Training Random Forest Classifier...")

rf_cls = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

rf_cls.fit(X_train, y_train_cls)
y_pred_cls = rf_cls.predict(X_test)

accuracy = accuracy_score(y_test_cls, y_pred_cls)

class_names = ["Green", "Yellow", "Red"]

cm = confusion_matrix(
    y_test_cls,
    y_pred_cls,
    labels=class_names,
)

print("\nCLASSIFICATION PERFORMANCE")
print("-" * 40)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification report:")
print(
    classification_report(
        y_test_cls,
        y_pred_cls,
        labels=class_names,
        zero_division=0,
    )
)

# ============================================================
# STEP 7: VISUALIZATION AND FEATURE IMPORTANCE
# ============================================================

print("\n[7/8] Creating evaluation plots...")

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": rf_reg.feature_importances_,
}).sort_values("Importance", ascending=False)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

# Feature importance chart
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].barh(
    importance_df["Feature"],
    importance_df["Importance"],
)
axes[0].invert_yaxis()
axes[0].set_xlabel("Importance")
axes[0].set_title("Random Forest Feature Importance")
axes[0].grid(alpha=0.25, axis="x")

cumulative = np.cumsum(importance_df["Importance"].to_numpy())

axes[1].plot(
    range(1, len(cumulative) + 1),
    cumulative,
    marker="o",
)
axes[1].axhline(0.90, linestyle="--", label="90% threshold")
axes[1].set_xlabel("Number of Features")
axes[1].set_ylabel("Cumulative Importance")
axes[1].set_title("Cumulative Feature Importance")
axes[1].legend()
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Prediction quality and confusion matrix
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

axes[0, 0].scatter(
    y_test_reg,
    y_pred_reg,
    alpha=0.3,
    s=10,
)

min_val = min(y_test_reg.min(), y_pred_reg.min())
max_val = max(y_test_reg.max(), y_pred_reg.max())

axes[0, 0].plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
)

axes[0, 0].set_xlabel("Actual Congestion")
axes[0, 0].set_ylabel("Predicted Congestion")
axes[0, 0].set_title(f"Actual vs Predicted (R2={r2:.4f})")
axes[0, 0].grid(alpha=0.25)

residuals = y_test_reg.to_numpy() - y_pred_reg

axes[0, 1].scatter(y_pred_reg, residuals, alpha=0.3, s=10)
axes[0, 1].axhline(0, color="red", linestyle="--")
axes[0, 1].set_xlabel("Predicted Congestion")
axes[0, 1].set_ylabel("Residual")
axes[0, 1].set_title("Residual Analysis")
axes[0, 1].grid(alpha=0.25)

axes[1, 0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
axes[1, 0].axvline(0, color="red", linestyle="--")
axes[1, 0].set_xlabel("Prediction Error")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Error Distribution")
axes[1, 0].grid(alpha=0.25)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[1, 1],
)

axes[1, 1].set_xlabel("Predicted Class")
axes[1, 1].set_ylabel("Actual Class")
axes[1, 1].set_title(f"Confusion Matrix (Accuracy={accuracy:.3f})")

plt.tight_layout()
plt.savefig("prediction_quality.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# STEP 8: SAVE MODELS AND RESULTS
# ============================================================

print("\n[8/8] Saving models and evaluation results...")

with open("congestion_regression_rf.pkl", "wb") as file:
    pickle.dump(rf_reg, file)

with open("congestion_classification_rf.pkl", "wb") as file:
    pickle.dump(rf_cls, file)

with open("feature_config.pkl", "wb") as file:
    pickle.dump(
        {
            "feature_cols": feature_cols,
            "label_encoder": label_encoder,
        },
        file,
    )

# Save test predictions for inspection.
results_df = pd.DataFrame({
    "actual_congestion": y_test_reg.to_numpy(),
    "predicted_congestion": y_pred_reg,
    "actual_class": y_test_cls.to_numpy(),
    "predicted_class": y_pred_cls,
    "tech_node": df.iloc[test_idx]["tech_node"].to_numpy(),
    "layout_id": df.iloc[test_idx]["layout_id"].to_numpy(),
})

results_df.to_csv("training_test_predictions.csv", index=False)

metrics = {
    "samples": len(df),
    "training_samples": len(train_idx),
    "testing_samples": len(test_idx),
    "features": len(feature_cols),
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "mape_percent": mape,
    "r2": r2,
    "cross_validation_r2_mean": cv_scores.mean(),
    "cross_validation_r2_std": cv_scores.std(),
    "classification_accuracy": accuracy,
}

pd.DataFrame([metrics]).to_csv("training_metrics.csv", index=False)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nSaved files:")
for filename in [
    "congestion_regression_rf.pkl",
    "congestion_classification_rf.pkl",
    "feature_config.pkl",
    "congestion_distribution.png",
    "feature_importance.png",
    "prediction_quality.png",
    "training_test_predictions.csv",
    "training_metrics.csv",
]:
    print(f"  - {filename}")

print("\nTop 3 features:")
print(importance_df.head(3).to_string(index=False))

print("\nFinal metrics:")
for key, value in metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

print("\nAll done. Models are ready for testing.")
