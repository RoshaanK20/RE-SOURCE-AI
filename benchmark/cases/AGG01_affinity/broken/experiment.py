from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.8, random_state=3)
agg = AgglomerativeClustering(n_clusters=4, affinity="manhattan", linkage="average")
labels = agg.fit_predict(X)
print("ari", adjusted_rand_score(y, labels))
