"""
K-means image simplifier.

This script loads an RGB image and reduces its color palette using
K-means clustering. The result is a simplified image with K colors.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# ======================
# Constants
# ======================
K = 20
MAX_ITER = 15
THRESHOLD = 25
SEED = 42

IMAGE_PATH = "momiji.jpg"


def validate_rgb_image(img: np.ndarray) -> None:
    """Raise a ValueError if the image is not a 3-channel RGB image."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("This script uses only color RGB images with no transparency.")


def select_random_centroids(img_f: np.ndarray, h: int, w: int, k: int):
    """Select k random pixels from the image as initial centroids."""
    ys = np.random.randint(0, h, size=k)
    xs = np.random.randint(0, w, size=k)
    centroids = img_f[ys, xs]
    return centroids


def compute_distances(img_f: np.ndarray, centroids: np.ndarray, h, w, k) -> np.ndarray:
    """Compute squared Euclidean distances from each pixel to each centroid."""
    distances = np.zeros((h, w, k), dtype=np.float32)
    for i in range(K):
        distances[:, :, i] = ((img_f - centroids[i]) ** 2).sum(axis=2)
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
            new_centroids[i] = select_random_centroids(img_f, 1)[0]
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
    centroids = centroids.round()
    new_img = img.copy()
    for i in range(k):
        new_img[labels == i] = centroids[i]
    return new_img
    

def main(k, max_iter, threshold, seed, image_path):
    """Read and validate the image"""
    img = io.imread(image_path)
    validate_rgb_image(img)
    img_f = img.astype(np.float32)
    
    """Plot the input image"""
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    
    """Initialize"""
    np.random.seed(seed)
    h, w = img.shape[:2]
    centroids = select_random_centroids(img_f, h, w, k)
    
    for iteration in range(0,max_iter):
        distances = compute_distances(img_f, centroids, h, w, K)
        labels = assign_labels(distances)
        new_centroids = recompute_centroids(img_f, labels, K)
        converged = check_convergence(new_centroids, centroids, THRESHOLD, iteration)
        centroids = new_centroids
        
        if converged:
            break
        
    new_img = apply_centroids(img, labels, new_centroids, K)

    """New the output image"""
    plt.imshow(new_img)
    plt.axis("off")
    plt.show()
    
main(K, MAX_ITER, THRESHOLD, SEED, IMAGE_PATH)
