import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle
import csv
import sys
import torch
import pretrainedmodels.utils as utils
import pretrainedmodels
from io import BytesIO
from PIL import Image
import requests

data_path = "/mnt/data/SCALABLE/HARRISON/pca/pca_130_new.csv"
tag_path = "/mnt/data/SCALABLE/HARRISON/pca/tag_list_clean.csv"


#Leggo dataset e carico dati training
arr = pd.read_csv(data_path, sep=",", header=None, index_col=None, dtype="float16")
arr = arr.to_numpy()

points_centroids_map = pickle.load(open('points_centroids_map.pickle', 'rb'))
centroids = pickle.load(open('centroids.pickle', 'rb'))
assignments = pickle.load(open('assignments.pickle', 'rb'))

pca_model = pickle.load(open('pca_model.pickle', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))



hashtags = []
with open(tag_path, "r") as csvFile: 
    reader = csv.reader(csvFile,delimiter = " ")
    for row in reader:
        hashtags.append(row)

'''
Loads pretrained alexnet model
'''
load_img = utils.LoadImage()
torch.set_printoptions(precision=20)

model_name = 'alexnet'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

tf_img = utils.TransformImage(model)

'''
given image file or image url, returns alexnet features vector
'''
def get_vector(image_name):
    if image_name.startswith("http"):
        response = requests.get(image_name)
        input_img = Image.open(BytesIO(response.content))

    else:
        input_img = load_img(image_name)
    input_tensor = tf_img(input_img)
    input_tensor = input_tensor.unsqueeze(0)
    input = torch.autograd.Variable(input_tensor, requires_grad=False)
    output_features = model.features(input).view(4096)
    return output_features.data.numpy()

'''
Compute all distances between one point and a list of points
'''
def get_distances(point, points):
    sP = points
    pA = point
    return np.linalg.norm(sP - pA, ord=2, axis=1.)  # 'distances' is a list

'''
Convert from array of indexes to array of points
'''

def indexlist_to_points(lst,dataset):
    points = np.empty([len(lst),dataset.shape[1]])
    for index, val in enumerate(lst):
        points[index] = dataset[val]
    return points

'''
inputs: image name and the k similar images wanted
'''
image_name = "https://r.hswstatic.com/w_907/gif/tesla-cat.jpg"
k = 15

#The vector is scaled and then pca reduced
vector = get_vector(image_name).reshape(1,-1)
sv = scaler.transform(vector)
sample = pca_model.transform(sv)


closest_centroid = np.nanargmin(get_distances(sample, centroids))
print("Closest centroid index:",closest_centroid)
centroid_points = indexlist_to_points(points_centroids_map[closest_centroid], arr)

nearest_k_images = np.argsort(get_distances(sample,centroid_points))[:k]

nearest_samples = []
for s in nearest_k_images:
    nearest_samples.append(points_centroids_map[closest_centroid][np.asscalar(s)])
    
hash_dict = {}
for n in nearest_samples:
    for h in hashtags[n]:
        if h in hash_dict:
            hash_dict[h] = hash_dict[h] + 1
        else:
            hash_dict[h] = 1

sorted_tags = sorted(hash_dict.items(), key=lambda kv: kv[1], reverse=True)            


for tag in sorted_tags:
    print(tag[0] + " " + str(tag[1]) )
