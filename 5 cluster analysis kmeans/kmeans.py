import numpy as np
import time
import matplotlib.pyplot as plt


# calculate Euclidean distance
def euclDistance(v1, v2) -> float:
    return np.sqrt(np.sum(np.power(v2 - v1, 2)))


# init centroids with random samples
def init_centroids(arr, k):
    num_samples, dim = arr.shape
    centroids = np.zeros((k, dim))

    for i in range(k):
        index = int(np.random.uniform(0, num_samples))
        centroids[i, :] = arr[index, :]
    return centroids


def kmeans(arr, k):
    num_samples = arr.shape[0]  #
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    cluster_assignment = np.zeros((num_samples, 2))
    cluster_changed = True

    ## step 1: init centroids
    centroids = init_centroids(arr, k)

    while cluster_changed:
        cluster_changed = False

        for i in range(num_samples):
            min_dist = 100000.0
            min_index = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], arr[i, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j

                    ## step 3: update its cluster
            if cluster_assignment[i, 0] != min_index:
                cluster_changed = True
                cluster_assignment[i, :] = min_index, min_dist ** 2  #

        ## step 4: update centroids
        for j in range(k):
            points_in_cluster = arr[np.nonzero(cluster_assignment[:, 0] == j)[0]]
            centroids[j, :] = np.mean(points_in_cluster, axis=0)

    return centroids, cluster_assignment
