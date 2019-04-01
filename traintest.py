import pandas as pd
import numpy as np
from matplotlib import pyplot
from libKMCUDA import kmeans_cuda
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv
import argparse

import pickle

def get_distances(point, points):
    sP = points
    pA = point
    return np.linalg.norm(sP - pA, ord=2, axis=1.)  # 'distances' is a list

def indexlist_to_points(lst,dataset):
    points = np.empty([len(lst),dataset.shape[1]])
    for index, val in enumerate(lst):
        points[index] = dataset[val]
    return points

def train(X,k = 169):
    scaler = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scaler.fit_transform(X)
    print("DATASET NORMALIZED")
    #Fitting the PCA algorithm with our Data
    pca_model = PCA(n_components=130).fit(data_rescaled)
    pca_arr = pca_model.transform(data_rescaled)
    pca_arr = pca_arr.astype('float32')
    #ADD 
    centroids, assignments = kmeans_cuda(
    pca_arr, k, metric="L2", verbosity=1, seed=3)
    points_centroids_map = {x: [] for x in range(0,169)}
    for index, item in enumerate(assignments):
        points_centroids_map[item].append(index)
    return scaler, pca_model, centroids, points_centroids_map

def getTags(sample,k,v,centroids, points_centroids_map, train_arr, hashtags):
    closest_centroid = np.nanargmin(get_distances(sample, centroids))
    #print("closest centroid:",closest_centroid)
    centroid_points = indexlist_to_points(points_centroids_map[closest_centroid], train_arr)
    distances = get_distances(sample,centroid_points)
    #print(distances)
    nearest_k_images = np.argsort(distances)[:k]
    #print(nearest_k_images)
    #print(nearest_k_images)
    #for s in nearest_k_images:
    #    nearest_samples.append(points_centroids_map[closest_centroid][np.asscalar(s)])

    hash_dict = {}
    for n in nearest_k_images:
        nearest_sample = points_centroids_map[closest_centroid][np.asscalar(n)]
        for h in hashtags[nearest_sample]:
            if h in hash_dict:
                hash_dict[h] = hash_dict[h] + 1/distances[n]
            else:
                hash_dict[h] = 1/distances[n]

    sorted_tags = sorted(hash_dict.items(), key=lambda kv: kv[1], reverse=True)
    #print(sorted_tags)
    return sorted_tags[:v]
    #for tag in sorted_tags:
    #    print(tag[0] + " " + str(tag[1]) 
          
def compareResults(predict, groundtruth):
    #print(predict, groundtruth)
    precision = len(np.intersect1d(predict,groundtruth)) / len(predict)
    recall = len(np.intersect1d(predict,groundtruth)) / len(groundtruth)
    accuracy = 1 if len(np.intersect1d(predict,groundtruth)) != 0 else 0
    return precision, recall, accuracy
    
def test(X,y,k,v,scaler, pca_model, centroids, points_centroids_map, train_arr, hashtags):
    avg_precision = []
    avg_recall = []
    avg_accuracy = []
    for index, val in enumerate(X):
        val = val.reshape(1,-1)
        sv = scaler.transform(val)
        pcav = pca_model.transform(sv)
        result = getTags(pcav, k,v, centroids, points_centroids_map, train_arr, hashtags)
        precision, recall, accuracy = compareResults(result, y[index])
        #print(metrics)
        avg_precision.append(precision)
        avg_recall.append(recall)
        avg_accuracy.append(accuracy)
    
    avg_precision = np.mean(avg_precision)
    avg_recall = np.mean(avg_recall)
    avg_accuracy = np.mean(avg_accuracy)
    return avg_precision , avg_recall , avg_accuracy


parser = argparse.ArgumentParser(description='A hashtag recommender system based on k-means, mini-batch fast k-means and a deep learning feature extraction phase.')
parser.add_argument('--train', dest='train', action='store_true', help="Computes the training using the chosen clustering algoritm, omit to just do the test phase")
parser.add_argument('--clustering', '-c', dest='clust', choices=['minibatch','kmcuda'], default='kmcuda')
parser.set_defaults(train=False)
args = parser.parse_args()

hashtags = []
with open("/mnt/data/SCALABLE/HARRISON/pca/tag_list_clean.csv", "r") as csvFile: 
    reader = csv.reader(csvFile,delimiter = " ")
    for row in reader:
        hashtags.append(np.asarray(row))

hashtags = np.asarray(hashtags)

if args.train is True:
    full_csv = "../data/SCALABLE/HARRISON/pca/full.csv"
    df = pd.read_csv(full_csv, sep=",", header=None, index_col=0, dtype="float32")
    print("DATASET LOADED")


    X_train, X_test, y_train, y_test = train_test_split(df, hashtags, test_size=0.10, random_state=42)

    scaler, pca_model, centroids, points_centroids_map = train(X_train, 162)    
    #Testing after traing:
    avg_precision, avg_recall, avg_accuracy  = test(X_test, y_test, 15,10, scaler, pca_model, centroids, points_centroids_map, df,hashtags)
    print("Precision, recall, accuracy:", avg_precision, avg_recall, avg_accuracy)
    #TODO SAVE ALL MODEL TO DISK
else:
    X_test = pickle.load(open('./data/dataset/X_test.pickle', 'rb'))
    y_test = pickle.load(open('./data/dataset/y_test.pickle', 'rb'))
    scaler = pickle.load(open('./data/kmcuda/scaler.pickle', 'rb'))
    pca_model = pickle.load(open('./data/kmcuda/pca_model.pickle', 'rb'))
    centroids = pickle.load(open('./data/kmcuda/centroids.pickle', 'rb'))
    final_X = pickle.load(open('./data/kmcuda/final_X.pickle', 'rb'))
    points_centroids_map = pickle.load(open('./data/kmcuda/points_centroids_map.pickle', 'rb'))
    avg_precision, avg_recall, avg_accuracy  = test(X_test.to_numpy() ,y_test, 50 , 1 , scaler, pca_model, centroids, points_centroids_map, final_X, hashtags)
    print("Precision, recall, accuracy:", avg_precision*100, avg_recall*100, avg_accuracy*100)

