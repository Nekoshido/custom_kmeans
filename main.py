import kmeans


if __name__ == "__main__":
    P = [(1.1, 2.5, 3, 4, 5, 6, 7, 8, 9), (3.4, 1.9, 3, 2, 1, 2, 3, 6, 8)]
    clusters, result = kmeans.kmeans(P, 2, True)
