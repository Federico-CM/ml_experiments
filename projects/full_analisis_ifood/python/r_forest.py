from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from process_data import load_and_clean_data

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support
)

# ------------------------------------------------------------
# 1. Load and clean the dataset
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ml_project1_data.csv"

df = load_and_clean_data(CSV_PATH)
print("Data was cleaned and processed")

# ------------------------------------------------------------
# 2. Create numeric feature from customer date
# ------------------------------------------------------------
df["Customer_Tenure_Days"] = (
    df["Dt_Customer"].max() - df["Dt_Customer"]
).dt.days

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
# 5. Split data into training and test sets
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# 6. Train random forest model
# ------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 7. Make predictions
# ------------------------------------------------------------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 8. Basic evaluation
# ------------------------------------------------------------
print("RANDOM FOREST RESULTS")

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
# 9. Compare different probability thresholds
# ------------------------------------------------------------
print("THRESHOLD COMPARISON")

threshold_results = []

for threshold in [0.50, 0.40, 0.30, 0.20]:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred_threshold,
        average="binary",
        zero_division=0
    )

    accuracy = accuracy_score(y_test, y_pred_threshold)

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

threshold_df = pd.DataFrame(threshold_results)
print(threshold_df)

# ------------------------------------------------------------
# 10. Repeated cross-validation using ROC AUC and PR AUC
# ------------------------------------------------------------
print("REPEATED CROSS-VALIDATION")

repeated_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

roc_auc_scores = cross_val_score(
    model,
    X,
    y,
    cv=repeated_cv,
    scoring="roc_auc"
)

pr_auc_scores = cross_val_score(
    model,
    X,
    y,
    cv=repeated_cv,
    scoring="average_precision"
)

print("\nRepeated 5-Fold ROC AUC Scores:")
print(roc_auc_scores)

print("\nMean ROC AUC:", roc_auc_scores.mean())
print("Standard Deviation ROC AUC:", roc_auc_scores.std())

print("\nRepeated 5-Fold PR AUC Scores:")
print(pr_auc_scores)

print("\nMean PR AUC:", pr_auc_scores.mean())
print("Standard Deviation PR AUC:", pr_auc_scores.std())

# ------------------------------------------------------------
# 11. Feature importances
# ------------------------------------------------------------
importance_table = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("TOP 10 FEATURE IMPORTANCES")
print(importance_table.head(10))
