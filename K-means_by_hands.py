# Whereas there is a library function for K-means in python, I am going to implement K-means without library function.
import numpy as np
import pandas as pd


def csv_to_array(filepath):
    # read csv, ignoring index column
    df = pd.read_csv(filepath, index_col=0)
    return df.values

# K number of means with 784 dimensions
class Kmeans:
    def __init__(self, data_path="./MNIST_train", k=10, dim=784):
        self.k = k
        self.dim = dim
        self.train_set = csv_to_array(data_path) / 255.0  # squash; range from 0 to 1

        # make a dict - {nth dimension: {index: dimension centers(random number at first)} * 10}
        # We need 10 small dicts for each dimensions because k = 10
        self.dimension_centers = self.initialize_dimension_centers()

        self.cluster_vectors = self.build_cluster_vectors()

    def initialize_dimension_centers(self):
        dimension_centers = {}
        for dim in range(self.dim):
            cluster_dict = {}
            for cluster_idx in range(self.k):
                cluster_dict[cluster_idx] = np.random.rand()
            dimension_centers[dim] = cluster_dict
        return dimension_centers

    # will apply cosine_similarity
    def cosine_similarity(self, arr1, arr2):
        numerator = np.dot(arr1, arr2)
        denominator = np.linalg.norm(arr1) * np.linalg.norm(arr2)
        return numerator / denominator if denominator != 0 else 0

    # make a new vector that contains center points of each pixel of each group
    def build_cluster_vectors(self):
        cluster_vectors = []
        for cluster_idx in range(self.k):
            vec = np.array([self.dimension_centers[d][cluster_idx] for d in range(self.dim)])
            cluster_vectors.append(vec)
        return cluster_vectors
    
    def update_centers(self, assignments):
        cluster_sums = {
            dim: {k: 0.0 for k in range(self.k)} for dim in range(self.dim)
        }
        cluster_counts = {
            dim: {k: 0 for k in range(self.k)} for dim in range(self.dim)
        }

        for i, x in enumerate(self.train_set):
            cluster = assignments[i]
            for dim in range(self.dim):
                cluster_sums[dim][cluster] += x[dim]
                cluster_counts[dim][cluster] += 1


        for dim in range(self.dim):
            for k in range(self.k):
                count = cluster_counts[dim][k]
                if count > 0:
                    avg = cluster_sums[dim][k] / count
                    self.dimension_centers[dim][k] = avg
                else:
                    self.dimension_centers[dim][k] = np.random.rand()
        
        self.cluster_vectors = self.build_cluster_vectors()


    # a labeling function to define the number that MNIST implies
    def add_label(nparr):
        pass