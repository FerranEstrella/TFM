import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root / "src"))

import SpectralClustering 

W = SpectralClustering.read_sc_matrix()

#SpectralClustering.spectral_clustering_indicators_vs_k(W, kmax=80, plot_name="indicators_vs_k_80.png")
#SpectralClustering.spectral_clustering_indicators_vs_k(W, kmax=8, plot_name="indicators_vs_k_8")


labels, W_clusters, indicators = SpectralClustering.spectral_clustering(W, 4, compute_calinski=False, compute_silhouette=False, compute_modularity=False, plot=False, save=True, norm=True)


df = SpectralClustering.read_aal("aal.txt")
SpectralClustering.plot_clusters_3d_brain(df, labels)
