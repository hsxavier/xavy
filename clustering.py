import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


###############
### k-Means ###
###############


def try_many_kmeans(k_min, k_max, X):
    """
    Apply k-means to `X` with k in the range `k_min` to `k_max` .
    
    Returns
    -------
    
    n_cluster : list of ints
        The number of clusters in each try (i.e. the ks).
    
    cluster_arr : list of arrays
        Each array contains the examples' labels for each k.
        
    inertia_arr : list of floats
        Each entry is the inertia for each k.
        
    score_arr : list of floats
        Each entry is the silhouette score for each k.
    """
    n_cluster   = range(k_min, k_max + 1)
    cluster_arr = []
    score_arr   = []
    inertia_arr = []
    for k in n_cluster:
        # Fit k-means:
        kmeans  = KMeans(n_clusters=k)
        cluster = kmeans.fit_predict(X)
        # Compute silhouette score:
        score   = silhouette_score(X, cluster)

        cluster_arr.append(cluster)
        inertia_arr.append(kmeans.inertia_)
        score_arr.append(score)
    
    return list(n_cluster), cluster_arr, inertia_arr, score_arr


def plot_silhouettes_for_cluster(k, silhouettes, cluster, offset=0, color='lightblue'):
    """
    Given a list of silhouette coefficients `silhouettes` and cluster labels `cluster`, 
    plot cluster `k` example's positions, ordered by their coefficients, as a function 
    of the coefficients. The position is offset by `offset` and the plot is filled with 
    `color`.
    """
    silhouettes_cluster = sorted(silhouettes[cluster == k])
    example_pos = np.arange(1, len(silhouettes_cluster) + 1) + offset
    pl.fill_betweenx(example_pos, 0, silhouettes_cluster, example_pos, color=color, alpha=0.7)
    pl.plot(silhouettes_cluster, example_pos, color='k')
    
    pl.text(0.05, np.mean(example_pos), str(k), fontsize=16, verticalalignment='center')
    
    return offset + len(silhouettes_cluster)


def plot_silhouettes_by_cluster(silhouettes, cluster, silhouette_score, color_map='coolwarm'):
    """
    Plot the `silhouettes` coefficients for each example, grouped by label `cluster`,
    and also show the overall `silhouette score`. Use `color_map` to color each cluster.
    """
    
    color_scheme = pl.get_cmap(color_map)
    cluster_idx  = np.unique(cluster)

    offset = 0
    for k in sorted(cluster_idx):
        color  = color_scheme(k / (len(cluster_idx) - 1))
        offset = plot_silhouettes_for_cluster(k, silhouettes, cluster, offset, color)
    
    pl.axvline(silhouette_score, color='k', linewidth=2, linestyle='--')
    
    pl.tick_params(labelsize=14)
    pl.xlabel('Silhouette', fontsize=14)
    #pl.ylabel('Example pos.\n(by cluster and coef.)', fontsize=14)



def plot_kmeans_metrics(n_cluster, inertias, silhouettes):
    """
    Given a list of number of clusters `n_cluster` (i.e. ks), a list 
    of `inertias` and `silhouettes` for each k, plot both metrics.
    """
    # Inertia plot:
    pl.subplot(1,2,1)
    pl.plot(n_cluster, inertias, marker='o', color='royalblue')
    pl.tick_params(labelsize=14)
    pl.xlabel('# clusters', fontsize=14)
    pl.ylabel('Inertia', fontsize=14)

    # Inertia plot:
    pl.subplot(1,2,2)
    pl.plot(n_cluster, silhouettes, marker='o', color='chocolate')
    pl.tick_params(labelsize=14)
    pl.xlabel('# clusters', fontsize=14)
    pl.ylabel('Silhouette score', fontsize=14)
    
    pl.tight_layout()


def multiple_knife_plots(n_cluster, X, cluster_arr, score_arr):
    """
    Plot examples' silhouettes by cluster for multiple numbers of clusters.
    
    Input
    -----
    
    n_cluster : list of ints
        The number of clusters (k) in each fit.
        
    X : numpy 2D array
        The examples' features.
        
    cluster_arr : list of arrays
        Each entry are the examples' labels for a certain k.
        
    score_arr : list of floats
        The silhouette scores for the clustering.
    """
    n_rows = int(len(n_cluster) / 3) + 1
    pl.figure(figsize=(15, 4 * n_rows))

    for i in range(len(n_cluster)):
        pl.subplot(n_rows, 3, i + 1)
        silhouettes = silhouette_samples(X, cluster_arr[i])
        plot_silhouettes_by_cluster(silhouettes, cluster_arr[i], score_arr[i])
        pl.xlim([-0.05, 0.95])
        if i % 3 == 0:
            pl.ylabel('Example pos.\n(by cluster and coef.)', fontsize=14)
        else:
            pl.tick_params(labelleft=False)
    pl.subplots_adjust(wspace=0, hspace=0)


###############################
### Hierarchical clustering ###
###############################


from scipy.cluster.hierarchy import linkage, dendrogram

def plot_dendrogram(X, method='ward', metric='euclidean', optimal_ordering=False, p=7, truncate_mode='level', color_threshold=None,
                    get_leaves=True, orientation='top', labels=None, count_sort=False, distance_sort=False, show_leaf_counts=True, no_plot=False,
                    no_labels=False, leaf_font_size=None, leaf_rotation=None, leaf_label_func=None, show_contracted=False, link_color_func=None,
                    ax=None, above_threshold_color='gray', verbose=False):
    
    if metric == 'manhattan':
        metric = 'cityblock'
    
    if verbose == True:
        print('Computing linkage...')
    Z = linkage(X, method=method, metric=metric, optimal_ordering=optimal_ordering)
    
    if verbose == True:
        print('Plotting dendrogram...')
    R = dendrogram(Z, p=p, truncate_mode=truncate_mode, color_threshold=color_threshold, get_leaves=get_leaves, orientation=orientation, 
                   labels=labels, count_sort=count_sort, distance_sort=distance_sort, show_leaf_counts=show_leaf_counts, no_plot=no_plot, 
                   no_labels=no_labels, leaf_font_size=leaf_font_size, leaf_rotation=leaf_rotation, leaf_label_func=leaf_label_func, 
                   show_contracted=show_contracted, link_color_func=link_color_func, ax=ax, above_threshold_color=above_threshold_color)


def build_linkage_df(X, method='ward', metric='euclidean', optimal_ordering=False):
    """
    Build an agglometarive clustering linkage matrix and return it
    as a Pandas DataFrame. For more information, check the 
    scipy.cluster.hierarchy.linkage function's documentation.
    
    Parameters
    ----------
    X : array, (n_samples, n_features)
        Initial data points to cluster.
    method : str
        How to aggregate the distance between data points
        when computing the distance between clusters. It 
        can be 'single', 'complete', 'average', 'weighted', 
        'centroid', 'median' or 'ward'.
    metric : str
        How to compute the distance between data points.
        It can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, 
        'manhattan', ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, 
        ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, 
        ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, 
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, 
        ‘sqeuclidean’ or ‘yule’.
    optimal_ordering : bool
        Reorder the linkage matrix so the distance between the 
        sucessive leave clusters are minimal. This can slow 
        down the computations significantly.
    """
    
    if metric == 'manhattan':
        metric = 'cityblock'
    
    Z = linkage(X, method=method, metric=metric, optimal_ordering=optimal_ordering)
    
    linkage_df = pd.DataFrame(data=Z, columns=['child_1', 'child_2', 'distance', 'n_samples'])
    linkage_df['child_1']   = linkage_df['child_1'].astype(int)
    linkage_df['child_2']   = linkage_df['child_2'].astype(int)
    linkage_df['n_samples'] = linkage_df['n_samples'].astype(int)
    linkage_df['merge']     = len(X) + linkage_df.index
    
    return linkage_df