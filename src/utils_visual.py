import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def pca_explained_variance_bar(transformed_x, alpha=0.8):
    """
    Plots a bar chart of the explained variance of each principal component
    of a PCA-transformed dataset, and highlights the number of components
    needed to explain a certain fraction of the total variance.
    
    Parameters:
    transformed_x (numpy.ndarray): The PCA-transformed dataset.
    alpha (float): The fraction of the total variance to be explained.
    
    Returns:
    None
    """
    # Compute the explained variance of each principal component
    explained_variance = np.var(transformed_x, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Find the number of components needed to explain alpha fraction of the total variance
    n_components = np.argmax(cumulative_explained_variance_ratio >= alpha) + 1
    
    # Plot the explained variance of each component
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, c='orange')
    plt.axvline(x=n_components, c='red', linestyle='--')
    plt.title('Explained Variance of Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.show()
