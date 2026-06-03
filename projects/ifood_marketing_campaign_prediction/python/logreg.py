from pathlib import Path
import pandas as pd
import numpy as np

from process_data import load_and_clean_data

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


# ------------------------------------------------------------
# 1. Load and clean the dataset
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ml_project1_data.csv"

df = load_and_clean_data(CSV_PATH)
print("Data was cleaned and processed")


# ------------------------------------------------------------
# 2. Create numeric feature from date column
# ------------------------------------------------------------
df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
df = df.drop(columns=["Dt_Customer"])


# ------------------------------------------------------------
# 3. Define target and features
# ------------------------------------------------------------
y = df["Response"]
X = df.drop(columns=["Response"])


# ------------------------------------------------------------
# 4. One-hot encode categorical variables
# ------------------------------------------------------------
X = pd.get_dummies(X, drop_first=True)


# ------------------------------------------------------------
# 5. Train/test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 6. Scale features
# ------------------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# 7. Train logistic regression model
# ------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# ------------------------------------------------------------
# 8. Make predictions
# ------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]


# ------------------------------------------------------------
# 9. Evaluate model
# ------------------------------------------------------------
print("\nLOGISTIC REGRESSION RESULTS")

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("PR AUC:", average_precision_score(y_test, y_pred_proba))

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

print("\nConfusion Matrix:")
print(cm_df)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ------------------------------------------------------------
# 10. Threshold comparison
# ------------------------------------------------------------
thresholds = [0.5, 0.4, 0.3, 0.2]

threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred_threshold),
        "Precision": precision_score(y_test, y_pred_threshold, zero_division=0),
        "Recall": recall_score(y_test, y_pred_threshold, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred_threshold, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nTHRESHOLD COMPARISON")
print(threshold_df)


# ------------------------------------------------------------
# 11. Repeated cross-validation
#     Uses a pipeline so scaling happens correctly inside each fold
# ------------------------------------------------------------
cv_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

roc_auc_scores = cross_val_score(
    cv_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

pr_auc_scores = cross_val_score(
    cv_pipeline,
    X,
    y,
    cv=cv,
    scoring="average_precision"
)

print("\nREPEATED CROSS-VALIDATION")
print("5-Fold x 10 Repeats ROC AUC Scores:", roc_auc_scores)
print("Mean ROC AUC:", roc_auc_scores.mean())
print("Standard Deviation:", roc_auc_scores.std())
print("Minimum ROC AUC:", roc_auc_scores.min())
print("Maximum ROC AUC:", roc_auc_scores.max())

print("\n5-Fold x 10 Repeats PR AUC Scores:", pr_auc_scores)
print("Mean PR AUC:", pr_auc_scores.mean())
print("Standard Deviation:", pr_auc_scores.std())
print("Minimum PR AUC:", pr_auc_scores.min())
print("Maximum PR AUC:", pr_auc_scores.max())


# ------------------------------------------------------------
# 12. Inspect coefficients
# ------------------------------------------------------------
coef_table = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", ascending=False)

print("\nTop positive coefficients:")
print(coef_table.head(10))

print("\nTop negative coefficients:")
print(coef_table.tail(10))
