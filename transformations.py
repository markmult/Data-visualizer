import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import Isomap, TSNE, MDS
from sklearn.decomposition import PCA
import umap

def scaling_and_normalization(data, normalizations):
    """
    Apply different scaling and normalization methods to data based on selected methods.
    """
    results = []
    for norm in normalizations:
        transform = None
        if norm == 'Minmax':
            transform = preprocessing.MinMaxScaler().fit_transform(data)
        elif norm == 'Standard scaler':
            transform = preprocessing.StandardScaler().fit_transform(data)
        elif norm == 'Mean':
            X = data.to_numpy()
            transform = (X - np.mean(X, axis=0)) / (X.max(axis=0) - X.min(axis=0))
        elif norm == 'Unit vector':
            transform = preprocessing.normalize(data, axis=1)
        elif norm == 'Robust scaler':
            transform = preprocessing.RobustScaler().fit_transform(data)
        results.append(transform)
    return np.array(results)


def dimensional_reduction(data, algorithms, parameters=[]):
    """
    Transfer data into two dimensions with selected algorithms.
    """
    results = []
    for algoritmi in algorithms:
        transform = None
        if algoritmi == 'PCA':
            transform = PCA(n_components=2, random_state=0).fit_transform(data)
        elif algoritmi == 'MDS':
            transform = MDS(n_jobs=-1, random_state=0).fit_transform(data)
        elif algoritmi == 'Isomap':
            transform = Isomap(n_neighbors=parameters[0], n_components=2).fit_transform(data)
        elif algoritmi == 'UMAP':
            transform = umap.UMAP(n_neighbors=parameters[1], min_dist=parameters[2],metric='correlation').fit_transform(data)
        elif algoritmi == 't-SNE':
            transform = TSNE(n_components=2, random_state=0, perplexity=parameters[3]).fit_transform(data)
        results.append(transform)
    return np.array(results)

def transform_data(data, normalizations, algorithms, parameters=[]):
    """
    Transform data with selected scalinig, normalization and dimensionality reduction methods.
    Returns numpy array with transformed data.
    """
    normalization_results = scaling_and_normalization(data, normalizations)
    result_data = []

    for norm in normalization_results:
        transforms = dimensional_reduction(norm, algorithms, parameters)
        result_data.append(transforms)
    
    return np.array(result_data)
