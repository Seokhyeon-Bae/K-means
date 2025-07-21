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

        self.dimension_centers = self.initialize_dimension_centers()

        self.cluster_vectors = self.build_cluster_vectors()
    
    
    # make a dict - {nth dimension: {index: dimension centers(random number at first)} * 10}
    # We need 10 small dicts for each dimensions because k = 10
    def initialize_dimension_centers(self):
        dimension_centers = {}
        for dim in range(self.dim):
            cluster_dict = {}
            for cluster_idx in range(self.k):
                cluster_dict[cluster_idx] = np.random.rand()
            dimension_centers[dim] = cluster_dict
        return dimension_centers

    # will apply cosine_similarity, you can also use euclidean distance
    # but I think cosine similarity is more suitable for this case
    # because we are dealing with images, and images are more similar to each other
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
    
    def assign_clusters(self):
        labels = []
        for x in self.train_set:
            sims = [self.cosine_similarity(x, center) for center in self.cluster_vectors]
            best_cluster = np.argmax(sims)
            labels.append(best_cluster)
        return np.array(labels)
            
    def update_centers(self, labels):
        # Update each dimension's center value for each cluster
        for dim in range(self.dim):
            for k in range(self.k):
                values = [self.train_set[i][dim] for i in range(self.n_samples) if labels[i] == k]
                if values:
                    self.dimension_centers[dim][k] = np.mean(values)
                else:
                    self.dimension_centers[dim][k] = np.random.rand()
        self.cluster_vectors = self.build_cluster_vectors()

    def fit(self, max_iter=15):
        labels = self.assign_clusters()
        for _ in range(max_iter):
            prev_labels = labels.copy()
            self.update_centers(labels)
            labels = self.assign_clusters()
            if np.array_equal(prev_labels, labels):
                break
        self.labels = labels
        return labels
    
    # a labeling function to define the number that MNIST implies
    # I will use 10 examples for each cluster to define the label
    def label_cluster(np_array):
        pass