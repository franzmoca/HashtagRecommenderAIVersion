import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
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


def train_1(X, k_pca = 130, save_result = False):
    #Fitting the PCA algorithm with our Data
    try:
        pca_model = pickle.load(open("./data/pca_model"+str(k_pca)+".pickle","rb"))
        scaler = pickle.load(open('./data/scaler.pickle', 'rb'))
        pca_arr = pickle.load(open('./data/dataset/finalX'+str(k_pca)+'.pickle', 'rb'))
        print("REDUCED DATASET LOADED")
    except IOError:
        print("PCA AND SCALER NOT FOUND")
        scaler = MinMaxScaler(feature_range=[0, 1])
        data_rescaled = scaler.fit_transform(X)
        print("DATASET NORMALIZED")
        pca_model = PCA(n_components=k_pca).fit(data_rescaled)
        pca_arr = pca_model.transform(data_rescaled)
        print("DATASET REDUCED")
        pickle.dump(scaler,open('./data/scaler.pickle', 'wb'))
        pickle.dump(pca_model,open('./data/pca_model'+str(k_pca)+'.pickle', 'wb'))
        pickle.dump(pca_arr,open('./data/dataset/finalX'+str(k_pca)+'.pickle', 'wb'))


    pca_arr = pca_arr.astype('float32')
    return scaler, pca_model, pca_arr

def train_2(X_pca, k = 169, clust = "kmcuda", save_result = True):
    if clust == "kmcuda":
        from libKMCUDA import kmeans_cuda
        centroids, assignments = kmeans_cuda(pca_arr, k, metric="L2", verbosity=1, seed=3)
        points_centroids_map = {x: [] for x in range(0,169)}
        for index, item in enumerate(assignments):
            points_centroids_map[item].append(index)
    elif clust == "minibatch":
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=100,n_init=3, max_no_improvement=10, verbose=0, random_state=42)
        mbk.fit(X_pca)
        points_centroids_map = {x: [] for x in range(0,k+1)}
        for index, item in enumerate(mbk.labels_):
            points_centroids_map[item].append(index)
        
        centroids = mbk.cluster_centers_
    else:
        print("UNIMPLEMENTED CLUSTERING METHOD; " + clust)
        return
    
    if save_result:
        pickle.dump(centroids, open('./data/'+clust+'/centroids'+str(k)+'.pickle', 'wb'))
        pickle.dump(points_centroids_map, open('./data/'+clust+'/points_centroids_map'+str(k)+'.pickle', 'wb'))

    return centroids, points_centroids_map



def getTags(sample,k,v,centroids, points_centroids_map, train_arr, hashtags):
    closest_centroid = np.nanargmin(get_distances(sample, centroids))
    #print("closest centroid:",closest_centroid)
    centroid_points = indexlist_to_points(points_centroids_map[closest_centroid], train_arr)
    distances = get_distances(sample,centroid_points)
    inverse_distances = np.power(distances, -1)
    #print(distances)
    if k == -1:
        nearest_k_images = range(len(distances))
    else:
        nearest_k_images = np.argsort(distances)[:k]

    hash_dict = {}
    for n in nearest_k_images:
        nearest_sample = points_centroids_map[closest_centroid][n]
        for h in hashtags[nearest_sample]:
            try:
                hash_dict[h] = hash_dict[h] + inverse_distances[n]
            except KeyError:
                hash_dict[h] = inverse_distances[n]

    sorted_tags = sorted(hash_dict.items(), key=lambda kv: kv[1], reverse=True)
    #print(sorted_tags)
    return sorted_tags[:v]
          
def compareResults(predict, groundtruth):
    #print(predict, groundtruth)
    #Calculate presision@1 first
    top_hp = predict[:1]
    precision_1 = len(np.intersect1d(top_hp,groundtruth)) / len(top_hp)
    #precision = len(np.intersect1d(predict,groundtruth)) / len(predict)
    recall = len(np.intersect1d(predict,groundtruth)) / len(groundtruth)
    accuracy = 1 if len(np.intersect1d(predict,groundtruth)) != 0 else 0
    return precision_1, recall, accuracy
    
def test(X,y,k,v,scaler, pca_model, centroids, points_centroids_map, train_arr, hashtags):
    avg_precision = []
    avg_recall = []
    avg_accuracy = []
    X_norm = scaler.transform(X)
    X_pca = pca_model.transform(X_norm)
    for index, val in enumerate(X_pca):
        val = val.reshape(1,-1)
        #sv = scaler.transform(val)
        #pcav = pca_model.transform(sv)
        result = getTags(val, k,v, centroids, points_centroids_map, train_arr, hashtags)
        precision, recall, accuracy = compareResults(result, y[index])
        avg_precision.append(precision)
        avg_recall.append(recall)
        avg_accuracy.append(accuracy)
    
    avg_precision = np.mean(avg_precision)
    avg_recall = np.mean(avg_recall)
    avg_accuracy = np.mean(avg_accuracy)
    return avg_precision , avg_recall , avg_accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A hashtag recommender system based on k-means, mini-batch fast k-means and a deep learning feature extraction phase.')
    parser.add_argument('--train', dest='train', action='store_true', help="Computes the training using the chosen clustering algoritm, omit to just do the test phase")
    parser.add_argument('--clustering', '-c', dest='clust', choices=['minibatch','kmcuda'], default='kmcuda')
    parser.add_argument('--clusters', '-k_c', dest="k_clusters", default=162, type=int)
    parser.add_argument('--n_pca',dest='k_pca',default=130, type=int)
    parser.add_argument('--nearest_images', '-n_i', dest="n_images", default=-1, type=int)
    parser.add_argument('--top_hashtags', '-k_ht', dest="top_hashtags", default=10, type=int)

    parser.set_defaults(train=False)
    args = parser.parse_args()


    if args.train:
        df = pickle.load(open("./data/dataset/full.pickle",'rb'))
        hashtags = pickle.load(open('./data/dataset/ht.pickle','rb'))
        print("DATASET LOADED")

        
        X_train, X_test, y_train, y_test = train_test_split(df, hashtags, test_size=0.10, random_state=42)
        print("DATASET SPLITTED")


        scaler, pca_model, pca_arr = train_1(X_train.to_numpy(), k_pca=args.k_pca)
        print("PCA APPLIED")

        centroids, points_centroids_map = train_2(pca_arr,k=args.k_clusters, clust=args.clust)
        print("CLUSTERING DONE")


        #Testing after traing:
        avg_precision, avg_recall, avg_accuracy  = test(X_test.to_numpy(), y_test, args.n_images , args.top_hashtags, scaler, pca_model, centroids, points_centroids_map, pca_arr ,y_train)
        print(f"Precision@1: {avg_precision*100}, Recall@{args.top_hashtags}:  {avg_recall*100}, Accuracy@{args.top_hashtags}:  {avg_accuracy*100}")

    else:
        try:
        #Load test set
            X_test = pickle.load(open('./data/dataset/X_test.pickle', 'rb'))
            y_test = pickle.load(open('./data/dataset/y_test.pickle', 'rb'))

            final_X = pickle.load(open('./data/dataset/finalX'+str(args.k_pca)+'.pickle', 'rb'))
            final_Y = pickle.load(open('./data/dataset/y_train.pickle', 'rb'))

            #Load scaler and pca_model
            scaler = pickle.load(open('./data/scaler.pickle', 'rb'))
            pca_model = pickle.load(open('./data/pca_model'+str(args.k_pca)+'.pickle', 'rb'))

            #Load clusterings
            centroids = pickle.load(open('./data/'+str(args.clust)+'/centroids'+str(args.k_clusters)+'.pickle', 'rb'))
            points_centroids_map = pickle.load(open('./data/'+str(args.clust)+'/points_centroids_map'+str(args.k_clusters)+'.pickle', 'rb'))
            print("Data Loading Complete, starting testing ... ")
            avg_precision, avg_recall, avg_accuracy  = test(X_test.to_numpy() ,y_test, args.n_images , args.top_hashtags , scaler, pca_model, centroids, points_centroids_map, final_X, final_Y)
            print(f"Precision@1: {avg_precision*100}, Recall@{args.top_hashtags}:  {avg_recall*100}, Accuracy@{args.top_hashtags}:  {avg_accuracy*100}")
        except IOError as e:
            print("NEED TO TRAIN FIRST", e)
