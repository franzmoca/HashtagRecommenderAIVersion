import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import pickle


path = "./data/pca_130_new.csv"
arr = pd.read_csv(path, sep=",", header=None, index_col=None, dtype="float16")
arr = arr.to_numpy()

k = 162
mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=100,
                      n_init=3, max_no_improvement=10, verbose=0, random_state=42)
mbk.fit(arr)

#Dizionario con come chiave il centroide e come valore la lista di punti corrispondenti.
points_centroids_map = {x: [] for x in range(0,k+1)}
for index, item in enumerate(mbk.labels_):
  points_centroids_map[item].append(index)
pickle.dump(points_centroids_map, open('./data/minibatch/points_centroids_map.pickle', 'wb'))
pickle.dump(mbk.cluster_centers_, open('./data/minibatch/centroids.pickle', 'wb'))
print(mbk.cluster_centers_.shape)

