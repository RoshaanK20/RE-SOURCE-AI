from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

X, y = make_blobs(n_samples=600, centers=8, cluster_std=2.2, random_state=7)
km = KMeans(n_clusters=8, algorithm="lloyd", n_init=10, random_state=0)
labels = km.fit_predict(X)
print("ari", adjusted_rand_score(y, labels))
print("inertia", km.inertia_)
