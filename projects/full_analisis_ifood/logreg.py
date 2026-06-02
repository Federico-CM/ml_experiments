from pathlib import Path
from process_data import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ml_project1_data.csv"

df = load_and_clean_data(CSV_PATH)
print("Data was cleaned and processed")


######################################

from pathlib import Path
import pandas as pd
import numpy as np

from process_data import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
#    Logistic regression cannot use raw datetime values directly
# ------------------------------------------------------------
df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
df = df.drop(columns=["Dt_Customer"])

# ------------------------------------------------------------
# 3. Define target (y) and features (X)
#    Here we assume "Response" is the binary outcome to predict
# ------------------------------------------------------------
y = df["Response"]
X = df.drop(columns=["Response"])

# ------------------------------------------------------------
# 4. One-hot encode categorical variables
#    This converts text columns like Education / Marital_Status
#    into numeric dummy variables
# ------------------------------------------------------------
X = pd.get_dummies(X, drop_first=True)

# ------------------------------------------------------------
# 5. Split data into training and test sets
#    stratify=y preserves class balance across train/test
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
#    Scaling is recommended for logistic regression,
#    especially when variables are on very different scales
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 7. Train the logistic regression model
# ------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# 8. Make predictions
# ------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# ------------------------------------------------------------
# 9. Evaluate the model
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
# 10. Inspect coefficients
#     Positive coefficient -> increases log-odds of Response=1
#     Negative coefficient -> decreases log-odds of Response=1
# ------------------------------------------------------------
coef_table = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", ascending=False)

print("\nTop positive coefficients:")
print(coef_table.head(10))

print("\nTop negative coefficients:")
print(coef_table.tail(10))

