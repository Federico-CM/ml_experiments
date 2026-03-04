"""
K-means image simplifier.

This script loads an RGB image and reduces its color palette using
K-means clustering. The result is a simplified image with K colors.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# ======================
# Constants
# ======================
K = 20
MAX_ITER = 15
THRESHOLD = 25 # this is squared distance
SEED = 1

# Directory where THIS script file lives
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "momiji.jpg"

# ======================
# I/O + Validation + Display
# ======================
def validate_data(
    img: np.ndarray,
    k: int,
    max_iter: int,
    threshold: float
) -> None:
    """
    Validate image and hyperparameters.

    Raises ValueError if any condition is invalid.
    """

    # Image validation
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            "This script supports only RGB images, no transparency."
        )

    h, w = img.shape[:2]
    n_pixels = h * w

    # Hyperparameter validation
    if not isinstance(k, int) or k <= 0:
        raise ValueError("K must be a positive integer.")

    if k > n_pixels:
        raise ValueError(
            f"K ({k}) cannot exceed number of pixels ({n_pixels})."
        )

    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("MAX_ITER must be a positive integer.")

    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValueError("THRESHOLD must be a float >= 0.")


def show_image(img: np.ndarray, title: str | None = None) -> None:
    """Display an image."""
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


# ======================
# K-means helpers
# ======================
def init_centroids(img_f: np.ndarray, k: int):
    """Select k random pixels from the image as initial centroids."""
    y, x = img_f.shape[:2]
    ys = np.random.randint(0, y, size=k)
    xs = np.random.randint(0, x, size=k)
    centroids = img_f[ys, xs]
    return centroids


def compute_distances(img_f: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distances from each pixel to each centroid."""
    diff = img_f[:, :, None, :] - centroids[None, None, :, :]
    distances = (diff * diff).sum(axis=3, dtype=np.float32)

    return distances


def assign_labels(distances: np.ndarray) -> np.ndarray:
    """Assign each pixel to the nearest centroid. Returns (H, W) int array."""
    labels = np.argmin(distances, axis=2)
    return labels


def recompute_centroids(img_f, labels, k):
    """Recalculate centroids for k clusters."""
    new_centroids = np.zeros((k, 3), dtype=np.float32)
    for i in range(k):
        mask = labels == i
        if not np.any(mask):  # if cluster is empty then re-seed
            new_centroids[i] = init_centroids(img_f, 1)[0]
        else:
            new_centroids[i] = img_f[mask].mean(axis=0)
    return new_centroids


def check_convergence(new_centroids, centroids, threshold, iteration):
    """Checks whether centroid movement is below a threshold."""
    max_movement = ((new_centroids - centroids) ** 2).sum(axis=1).max()

    if max_movement < threshold:
        print("Reached convergence at iteration:", iteration + 1)
        return True

    print("Finished iteration:", iteration + 1)
    return False


def apply_centroids(img, labels, centroids, k):
    """Replace pixels in img with their corresponding centroid values."""
    centroids = centroids.round().clip(0, 255).astype(img.dtype)

    new_img = img.copy()
    for i in range(k):
        new_img[labels == i] = centroids[i]
    return new_img


def main(k, max_iter, threshold, seed, image_path):
    """Execute the full K-means image simplification pipeline."""
    # Read and validate the image
    img = io.imread(image_path)
    validate_data(img, k, max_iter, threshold)
    img_f = img.astype(np.float32)

    # Plot the input image
    show_image(img, title="Input")

    # Initialize
    np.random.seed(seed)
    centroids = init_centroids(img_f, k)

    # Run the main loop
    for iteration in range(0, max_iter):
        distances = compute_distances(img_f, centroids)
        labels = assign_labels(distances)
        new_centroids = recompute_centroids(img_f, labels, k)
        converged = check_convergence(new_centroids, centroids, threshold, iteration)
        centroids = new_centroids

        if converged:
            break

    new_img = apply_centroids(img, labels, new_centroids, k)

    # Show the output image
    show_image(new_img, title="Output")


if __name__ == "__main__":
    main(K, MAX_ITER, THRESHOLD, SEED, IMAGE_PATH)
