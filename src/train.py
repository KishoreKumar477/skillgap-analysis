import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns

#  Load Data 
df = pd.read_csv(r"/data/processed/ml_job_features.csv")

SKILL_COLS = [c for c in df.columns if c not in [
    "job_id", "job_title", "company_name",
    "role_category", "experience_level", "url"
]]

X = df[SKILL_COLS]
y = df["role_category"]

#  Encode Labels 
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Compute weights to balance classes
sample_weights = compute_sample_weight("balanced", y_train)

#  MLflow Tracking 
mlflow.set_experiment("skill-gap-analyzer")

with mlflow.start_run(run_name="xgboost-baseline"):

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "early_stopping_rounds": 20
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Evaluate 
    y_pred = model.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    report = classification_report(y_test_labels, y_pred_labels)
    print("\nClassification Report:")
    print(report)

    #  Log to MLflow
    mlflow.log_params(params)
    
    accuracy = (y_pred == y_test).mean()
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")

    mlflow.xgboost.log_model(model, "model")

    #  Confusion Matrix 
    cm = confusion_matrix(y_test_labels, y_pred_labels,
                          labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("/data/processed/confusion_matrix.png")
    mlflow.log_artifact("/data/processed/confusion_matrix.png")
    print("Saved confusion_matrix.png")

    # SHAP Explainability 
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot — top skills driving predictions
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[:,:,3],
        X_test,
        feature_names=SKILL_COLS,
        max_display=20,
        show=False,
        plot_type="dot"
    )
    plt.title("SHAP — Top skills driving 'data_scientist' prediction")

    plt.tight_layout()
    plt.savefig(r"/data/processed/shap_summary.png", dpi=150, bbox_inches="tight")
    mlflow.log_artifact(r"/data/processed/shap_summary.png")
    print("Saved shap_summary.png")

    #  Save Model Artifacts 
    model.save_model(r"/data/processed/xgb_model.json")
    
    import pickle
    with open("/data/processed/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\nAll artifacts saved.")
    print(f"MLflow run complete. Run: mlflow ui to view.")
