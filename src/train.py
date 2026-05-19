import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV  # ADDED: for probability calibration
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ── Load Data ─────────────────────────────────────────────────────
df = pd.read_csv("data/processed/ml_job_features.csv")

SKILL_COLS = [c for c in df.columns if c not in [
    "job_id", "job_title", "company_name",
    "role_category", "experience_level", "url"
]]

X = df[SKILL_COLS]
y = df["role_category"]

# ── Encode Labels ─────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)

# ── Train/Test Split ──────────────────────────────────────────────
# CHANGED: added X_val, y_val split for calibration
# we need 3 sets: train (fit model), val (fit calibrator), test (evaluate)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ADDED: carve out 15% of remaining data as validation set for calibration
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Class Weights ─────────────────────────────────────────────────
sample_weights_train = compute_sample_weight("balanced", y_train)  # CHANGED: renamed for clarity

# ── MLflow Tracking ───────────────────────────────────────────────
mlflow.set_experiment("skill-gap-analyzer")

with mlflow.start_run(run_name="xgboost-calibrated"):  # CHANGED: updated run name

    params = {
        "n_estimators":     200,
        "max_depth":        6,
        "learning_rate":    0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "eval_metric":      "mlogloss",
        "random_state":     42,
    }

    # ── Train Base Model ──────────────────────────────────────────
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],   # CHANGED: eval on val set, not test set
        early_stopping_rounds=20,
        verbose=50
    )

    # ── Calibrate Probabilities ───────────────────────────────────
    # ADDED: raw XGBoost probabilities are overconfident
    # isotonic regression fits a monotonic function to fix this
    # cv='prefit' means we use the already-trained booster
    # result: 36% on a strong resume means genuinely uncertain, not model quirk
    calibrated_model = CalibratedClassifierCV(
        model, cv='prefit', method='isotonic'
    )
    calibrated_model.fit(X_val, y_val)
    print("Calibration complete.")

    # ── Evaluate using calibrated model ──────────────────────────
    # CHANGED: use calibrated_model for all predictions going forward
    y_pred        = calibrated_model.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    report = classification_report(y_test_labels, y_pred_labels)
    print("\nClassification Report:")
    print(report)

    # ── Cross Validation ──────────────────────────────────────────
    # ADDED: 5-fold CV makes accuracy claim defensible
    # single split can be lucky — CV gives mean ± std
    print("\nRunning 5-fold cross validation...")
    cv_model = xgb.XGBClassifier(**params)
    cv_scores = cross_val_score(
        cv_model, X, y_encoded,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy"
    )
    print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Baseline Comparison ───────────────────────────────────────
    # ADDED: compare against naive baselines so accuracy is meaningful
    majority_class_acc = float(pd.Series(y_encoded).value_counts().max() / len(y_encoded))
    random_acc = 1.0 / len(le.classes_)
    print(f"\nBaseline Comparison:")
    print(f"  Random classifier:  {random_acc:.3f}")
    print(f"  Majority class:     {majority_class_acc:.3f}")
    print(f"  Our model (CV):     {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Log to MLflow ─────────────────────────────────────────────
    mlflow.log_params(params)

    accuracy = float((y_pred == y_test).mean())
    mlflow.log_metric("accuracy",             accuracy)
    mlflow.log_metric("cv_accuracy_mean",     float(cv_scores.mean()))
    mlflow.log_metric("cv_accuracy_std",      float(cv_scores.std()))
    mlflow.log_metric("baseline_majority",    majority_class_acc)   # ADDED
    mlflow.log_metric("baseline_random",      random_acc)           # ADDED

    print(f"\nTest Accuracy:  {accuracy:.4f}")
    print(f"CV Accuracy:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # CHANGED: log calibrated model via pickle, not mlflow.xgboost
    # mlflow.xgboost only works with raw XGBoost, not sklearn wrapper
    mlflow.log_artifact("data/processed/xgb_model_calibrated.pkl") if os.path.exists("data/processed/xgb_model_calibrated.pkl") else None

    # ── Confusion Matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("data/processed/confusion_matrix.png")
    mlflow.log_artifact("data/processed/confusion_matrix.png")
    print("Saved confusion_matrix.png")

    # ── SHAP Explainability ───────────────────────────────────────
    # NOTE: SHAP uses raw model (not calibrated) — TreeExplainer
    # needs direct access to XGBoost booster internals
    print("\nComputing SHAP values...")
    explainer   = shap.TreeExplainer(model)   # UNCHANGED: intentionally uses raw model
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[:, :, 3],
        X_test,                        # CHANGED: was missing X_test argument
        feature_names=SKILL_COLS,
        max_display=20,
        show=False,
        plot_type="dot"
    )
    plt.title("SHAP — Top skills driving 'data_scientist' prediction")
    plt.tight_layout()
    plt.savefig("data/processed/shap_summary.png", dpi=150, bbox_inches="tight")
    mlflow.log_artifact("data/processed/shap_summary.png")
    print("Saved shap_summary.png")

    # ── Save Artifacts ────────────────────────────────────────────
    # CHANGED: save calibrated model as pickle for flask_app.py
    with open("data/processed/xgb_model_calibrated.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)
    print("Saved xgb_model_calibrated.pkl")

    # keep raw model for SHAP usage only
    model.save_model("data/processed/xgb_model.json")
    print("Saved xgb_model.json (raw, for SHAP only)")

    # save label encoder
    with open("data/processed/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\nAll artifacts saved.")
    print(f"MLflow run complete. Run: mlflow ui to view.")
