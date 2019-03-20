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

ks = {}
lastMinwss = 3420000

for k in range(21, 171):
  mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=100,
                      n_init=3, max_no_improvement=10, verbose=0)
  mbk.fit(arr)
  minwss = mbk.inertia_
  print(minwss)
if minwss < lastMinwss:
    lastMinwss = minwss
    ks['wss'] = k
#print(ks)
      #for (method, k) in ks:
#print(f'For {method} k is {k}')

  #print(silhouette_score(arr, mbk.labels_, metric="euclidean", sample_size=None, random_state=None))
