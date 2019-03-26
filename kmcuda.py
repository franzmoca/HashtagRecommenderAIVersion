import pandas as pd
import numpy as np
from matplotlib import pyplot
from libKMCUDA import kmeans_cuda
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle


full_csv = "/mnt/data/SCALABLE/HARRISON/pca/full.csv"
df = pd.read_csv(full_csv, sep=",", header=None, index_col=0)
print("DATASET LOADED")

scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(df)
print(data_rescaled)
print("DATASET NORMALIZED")
#Fitting the PCA algorithm with our Data
pca_model = PCA(n_components=130).fit(data_rescaled)
print("PCA FIT")

pickle.dump(pca_model, open('pca_model.pickle', 'wb'))
pickle.dump(scaler, open('scaler.pickle', 'wb'))
pca_arr = pca_model.transform(data_rescaled)
data = pd.DataFrame(pca_arr).to_csv("/mnt/data/SCALABLE/HARRISON/pca/pca_130_new.csv", header=None, index=None)

path = "/mnt/data/SCALABLE/HARRISON/pca/pca_130_new.csv"
arr = pd.read_csv(path, sep=",", header=None, index_col=None, dtype="float16")
arr = arr.to_numpy()

centroids, assignments = kmeans_cuda(arr, 169, metric="L2", verbosity=1, seed=3)

#Dizionario con come chiave il centroide e come valore la lista di punti corrispondenti.
points_centroids_map = {x: [] for x in range(0,169)}
for index, item in enumerate(assignments):
    points_centroids_map[item].append(index)

pickle.dump(points_centroids_map, open('points_centroids_map.pickle', 'wb'))
pickle.dump(centroids, open('centroids.pickle', 'wb'))
pickle.dump(assignments, open('assignments.pickle', 'wb'))





