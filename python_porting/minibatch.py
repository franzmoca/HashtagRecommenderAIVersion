import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import pickle


path = "./data/pca_130_new.csv"
arr = pd.read_csv(path, sep=",", header=None, index_col=None, dtype="float16")
arr = arr.to_numpy()
k = 169
mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=100,
                      n_init=3, max_no_improvement=10, verbose=0)
mbk.fit(arr)
print(len(mbk.cluster_centers_))
print(np.argwhere(np.isnan(mbk.cluster_centers_)))
print(silhouette_score(arr, mbk.labels_, metric="euclidean",
                                   sample_size=None, random_state=None))
