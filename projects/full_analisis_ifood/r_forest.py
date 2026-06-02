from pathlib import Path
import pandas as pd

from process_data import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------
# 1. Load and clean the dataset
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ml_project1_data.csv"

df = load_and_clean_data(CSV_PATH)
print("Data was cleaned and processed")

# ------------------------------------------------------------
# 2. Create a usable numeric feature from the date column
# ------------------------------------------------------------
df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
df = df.drop(columns=["Dt_Customer"])

# ------------------------------------------------------------
# 3. Define target (y) and features (X)
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
# 6. Train the random forest model
# ------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,      # number of trees
    max_depth=None,        # let trees grow fully unless you want to limit depth
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced"   # useful if Response classes are imbalanced
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 7. Make predictions
# ------------------------------------------------------------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 8. Evaluate the model
# ------------------------------------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))

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
# 9. Inspect feature importances
# ------------------------------------------------------------
importance_table = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 10 most important features:")
print(importance_table.head(10))
