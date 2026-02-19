# lda_iris.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Fit LDA (classifier)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Predict
    y_pred = lda.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # Optional: project to LDA space (max 2 components for 3 classes -> 2D)
    try:
        import matplotlib.pyplot as plt

        X_lda = lda.transform(X)  # shape: (n_samples, 2)
        plt.figure()
        for cls in sorted(set(y)):
            plt.scatter(X_lda[y == cls, 0], X_lda[y == cls, 1], label=target_names[cls])
        plt.title("Iris - LDA projection (2D)")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nmatplotlib not installed; skipping plot.")

if __name__ == "__main__":
    main()

