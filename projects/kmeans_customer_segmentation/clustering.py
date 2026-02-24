import os
import numpy as np
import matplotlib.pyplot as plt

# make it so that the directory can be set
# make it so that there is a message on where the output is produced

# ----------------------------
# 1) Generate synthetic data
# ----------------------------
np.random.seed(42)

n_per_cluster = 150

true_centers = np.array([
    [50,  35],
    [75,  55],
    [105, 60]
], dtype=float)

sd_income = 10
sd_spend  = 8

def make_cluster(center, n, sd_income, sd_spend):
    income = np.random.normal(center[0], sd_income, n)
    spending = np.random.normal(center[1], sd_spend, n)
    return np.column_stack([income, spending])

X = np.vstack([
    make_cluster(true_centers[0], n_per_cluster, sd_income, sd_spend),
    make_cluster(true_centers[1], n_per_cluster, sd_income, sd_spend),
    make_cluster(true_centers[2], n_per_cluster, sd_income, sd_spend),
])

# ----------------------------
# 2) Manual K-means
# ----------------------------
k = 3
max_iter = 30
tolerance = 1e-4

init_idx = np.random.choice(X.shape[0], size=k, replace=False)
centroids = X[init_idx].copy()

# Create output folder
output_dir = "kmeans_frames"
os.makedirs(output_dir, exist_ok=True)

for it in range(1, max_iter + 1):

    # --- Assignment ---
    distances = np.zeros((X.shape[0], k))
    for j in range(k):
        distances[:, j] = np.sum((X - centroids[j])**2, axis=1)

    cluster_assign = np.argmin(distances, axis=1)

    # --- Plot ---
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(
        X[:, 0], X[:, 1],
        c=cluster_assign,
        s=25,
        alpha=0.9
    )
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c=np.arange(k),
        marker="x",
        s=250,
        linewidths=3
    )

    plt.title(f"Iteration {it}")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (0-100)")
    plt.xlim(15, 140)
    plt.ylim(0, 100)

    # Save image
    filename = f"/home/proud/Desktop/work/AI/kmeans/iter_{it:02d}.png"
    #filename = os.path.join(output_dir, f"iter_{it:02d}.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    # --- Update ---
    new_centroids = np.zeros_like(centroids)
    for j in range(k):
        pts = X[cluster_assign == j]
        if len(pts) == 0:
            new_centroids[j] = X[np.random.randint(0, X.shape[0])]
        else:
            new_centroids[j] = pts.mean(axis=0)

    shift = np.sum((centroids - new_centroids)**2)

    if shift < tolerance:
        print(f"Converged at iteration {it}")
        centroids = new_centroids
        break

    centroids = new_centroids

print("\nFinal centroids:")
print(centroids)