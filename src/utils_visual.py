import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pca_explained_variance_bar(x, alpha=0.8):
    """
    Plot the explained variance of each principal component and the cumulative explained variance.

    Args:
        x: The data matrix (numpy.ndarray).
        alpha: The fraction of the total variance that we want to explain (float).

    Returns:
        None
    """
    # Compute the explained variance of each principal component
    explained_variance = np.var(x, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Find the number of components needed to explain alpha fraction of the total variance
    n_components = np.argmax(cumulative_explained_variance_ratio >= alpha) + 1
    
    # Plot the explained variance of each component
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, c='orange')
    plt.axvline(x=n_components, c='red', linestyle='--') # type: ignore
    plt.title('Explained Variance of Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.show()
