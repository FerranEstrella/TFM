import os
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from itertools import combinations
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score, adjusted_rand_score

def read_sc_matrix():
    root = Path(__file__).resolve().parent.parent
    data_folder = root / "data"
    with open(data_folder / "sc2017.dat", "r") as f:
        n = int(f.readline().strip())  
    
        edges = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                edges.append((int(parts[0]), int(parts[1]), float(parts[2])))


    W = np.zeros((n, n))

    for frm, to, weight in edges:
        W[frm, to] = weight
  
    return W


def laplacian_rw_spectrum(W,plot=False):
    n = W.shape[0]

    d = W.sum(axis=1)
    d_safe = np.where(d != 0, d, 1)
    D_inv = np.diag(1 / d_safe)

    L_rw = np.eye(n) - D_inv @ W
    
    evals, evecs = np.linalg.eig(L_rw)
    idx = np.argsort(evals)
    evals_ordered = evals[idx]
    evecs_ordered = evecs[:, idx]
    
    if plot:
        plt.figure(figsize=(8,5))
        plt.plot(range(1, n+1), evals_ordered, marker='o')
        plt.xlabel('i')
        plt.ylabel('Eigenvalue')
        plt.title('Random Walk Laplacian Spectrum')
        plt.grid(True)
        
        scripts_folder = Path(__file__).resolve().parent.parent / "scripts"
        plt.savefig(scripts_folder / "laplacian_rw_spectrum.png", dpi=300, bbox_inches='tight')
        plt.close() 
       

    return evals_ordered, evecs_ordered


def spectral_clustering(W, k, compute_calinski=True, compute_silhouette=True, compute_modularity=True, 
    plot=False, save=False, norm=False):

    n = W.shape[0]
    evals, evecs = laplacian_rw_spectrum(W,plot)
    X = evecs[:, :k].real
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=None)
    labels = kmeans.fit_predict(X)
    indicators = {}

    if compute_calinski:
        indicators['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    if compute_silhouette:
        indicators['silhouette'] = silhouette_score(X, labels)
    if compute_modularity:
        g = ig.Graph.Weighted_Adjacency(W.tolist(), mode=ig.ADJ_DIRECTED)
        indicators['modularity'] = g.modularity(labels, weights=g.es["weight"])
    
    W_clusters = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            nodes_i = np.where(labels == i)[0]
            nodes_j = np.where(labels == j)[0]
            if len(nodes_i) > 0 and len(nodes_j) > 0:
                if i!=j:
                    W_clusters[i, j] = W[np.ix_(nodes_i, nodes_j)].mean()
    
    if save:
        root = Path(__file__).resolve().parent.parent
        data_folder = root / "data"
        with open(data_folder / "clustered_data.dat", "w") as f:
            f.write(f"{k}\n")
            for i in range(k):
                for j in range(k):
                    if W_clusters[i, j] != 0:   
                        f.write(f"{i} {j} {W_clusters[i,j]}\n") 

    if norm:

        d = W_clusters.sum(axis=1)
        W_norm = W_clusters / d[:, np.newaxis]

        root = Path(__file__).resolve().parent.parent
        data_folder = root / "data"
        np.save(str(data_folder / "normalized_matrix_4cluster.npy"), W_norm)


    return labels, W_clusters, indicators


def spectral_clustering_ari(W, k, n_init=10):
    
    labels_list = []

    for _ in range(n_init):
        labels, _, _ = spectral_clustering(
            W, k=k,
            compute_calinski=False,
            compute_silhouette=False,
            compute_modularity=False
        )
        labels_list.append(labels)

    ari_values = [adjusted_rand_score(l1, l2) for l1, l2 in combinations(labels_list, 2)]
    ari_mean = np.mean(ari_values)

    return ari_mean





def spectral_clustering_indicators_vs_k(W, kmax, plot_name):

    k_values = range(2, kmax + 1)
    calinski_list = []
    silhouette_list = []
    modularity_list = []
    ARI_list = []

    for k in k_values:
        labels, W_clusters, indicators = spectral_clustering(
            W, k=k,
            compute_calinski=True,
            compute_silhouette=True,
            compute_modularity=True,
            plot=True
        )
        calinski_list.append(indicators.get('calinski_harabasz', np.nan))
        silhouette_list.append(indicators.get('silhouette', np.nan))
        modularity_list.append(indicators.get('modularity', np.nan))
        ARI_list.append(spectral_clustering_ari(W, k))

    plt.figure(figsize=(12, 4))

    plt.subplot(2, 2, 1)
    plt.plot(k_values, calinski_list, marker='o')
    plt.xlabel('k')
    plt.ylabel('Calinski-Harabasz')
    plt.title('Calinski-Harabasz vs k')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(k_values, ARI_list, marker='o', color='purple')
    plt.xlabel('k')
    plt.ylabel('Adjusted Rand Index')
    plt.title('Adjusted Rand Index vs k')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(k_values, silhouette_list, marker='o', color='orange')
    plt.xlabel('k')
    plt.ylabel('Silhouette')
    plt.title('Silhouette vs k')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(k_values, modularity_list, marker='o', color='green')
    plt.xlabel('k')
    plt.ylabel('Modularity Q')
    plt.title('Modularity vs k')
    plt.grid(True)

    plt.tight_layout()

    scripts_folder = Path(__file__).resolve().parent.parent / "scripts"
    plt.savefig(scripts_folder / plot_name, dpi=300, bbox_inches='tight')
    plt.close() 


    results = {
        'k_values': list(k_values),
        'calinski_harabasz': calinski_list,
        'silhouette': silhouette_list,
        'modularity': modularity_list,
        'adjusted_rand_index': ARI_list
    }

    return results




def read_aal(filename="aal.txt"):
    """
    Reads the AAL coordinate file.
    """
    root = Path(__file__).resolve().parent.parent
    data_folder = root / "data"
 
    df = pd.read_csv(
        data_folder / filename,
        sep=r"\s+",
        engine="python",
        skiprows=1,
        names=["id","name","x","y","z","lobe","hemi","index"]
    )

    return df


def compute_brain_hull(df):
    """
    Computes convex hull of the node coordinates.
    """
    points = df[["x","y","z"]].values
    hull = ConvexHull(points)
    return hull, points



def plot_clusters_3d_brain(df, C, plot_name="brain_clusters.png"):

    df = df.copy()
    df["cluster"] = C

    points = df[["x","y","z"]].values
    hull = ConvexHull(points)

    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            i=hull.simplices[:,0],
            j=hull.simplices[:,1],
            k=hull.simplices[:,2],
            opacity=0.3,
            color="lightgrey"
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker=dict(
                size=6,
                color=df["cluster"],
                colorscale="Turbo"
            ),
            text=df["name"]
        )
    )

    fig.update_layout(width=900, height=800)

    scripts_folder = Path(__file__).resolve().parent.parent / "scripts"
    scripts_folder.mkdir(exist_ok=True)

    fig.write_html(scripts_folder / "brain_clusters.html")
    print("Saving to:", scripts_folder)
