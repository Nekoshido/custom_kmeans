import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt


def euclidean_distance(a, b, axis=None):
    return np.linalg.norm(a - b, axis=axis)


def kmeans(P, n_clusters, visualization=False):
    data = np.array(list(zip(P[0], P[1])))
    centroids = np.array(
        list(zip(np.random.randint(np.amin(data, axis=0)[0], np.amax(data, axis=0)[0], size=n_clusters),
                 np.random.randint(np.amin(data, axis=0)[1], np.amax(data, axis=0)[1],
                                   size=n_clusters))), dtype=np.float32)
    centroids_old = np.zeros(centroids.shape)
    distances = np.zeros((data.shape[0], n_clusters))
    error = euclidean_distance(centroids, centroids_old)
    while error != 0:
        for i in range(n_clusters):
            distances[:, i] = euclidean_distance(data, centroids[i], 1)
        clusters = np.argmin(distances, axis=1)
        centroids_old = deepcopy(centroids)
        for i in range(n_clusters):
            centroids[i] = np.mean(data[clusters == i], axis=0)
        error = euclidean_distance(centroids, centroids_old)
    if visualization:
        plt.scatter(data[:, 0], data[:, 1], c='b', s=5)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='g', s=200)
        plt.show()
    return clusters, centroids
