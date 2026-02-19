# log_reg_breast_cancer.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
target_names = cancer.target_names  # ['malignant', 'benign']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.25, random_state=42, stratify=y
)

# Extract summary statistics for each feature and normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Normalize
X_test_scaled = scaler.transform(X_test)

# Fit logistic regression
clf = LogisticRegression(max_iter=200, solver="lbfgs")
clf.fit(X_train_scaled, y_train)

# Predict
y_pred = clf.predict(X_test_scaled)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print(
	"\nClassification report:\n",
	classification_report(y_test, y_pred, target_names=target_names),
)

 
