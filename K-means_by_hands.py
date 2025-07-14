# Whereas there is a library function for K-means in python, I am going to implement K-means without library function.
import numpy as np
import pandas as pd


def csv_to_array(filepath):
    # read csv, ignoring index column
    df = pd.read_csv(filepath, index_col=0)

    features = df.values  # shape: (60000, 784)

    return features

class Kmeans:
    def __init__(self):
    # make a center points array with 784 dimensions, and make a cluster with 10 center points array, since there are 10 numbers.
        dataset = "./MNIST_train"
        dataset = dataset / 255.0 # squash data into the range from 0 to 1
        self.cluster = [np.random.rand(784) for _ in range(10)]
        self.train_set = csv_to_array(dataset)

    # will apply cosine_similarity
    def cosine_similarity(arr1,arr2):
        numerator = np.dot(arr1, arr2)
        denominator = np.linalg.norm(arr1) * np.linalg.norm(arr2)
        return numerator / denominator if denominator != 0 else 0

    def center_point_setup(arr):
        arr
        pass
    
    def add_label(nparr):
        pass