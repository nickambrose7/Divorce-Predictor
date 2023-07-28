import numpy as np

class Project_StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class Project_MinMaxScaler:
    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt(np.sum((a - b)**2, axis=1))

class Project_KMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol  # Tolerance to decide if centroids have significantly changed
        np.random.seed(random_state)

    def init_centroids(self, X):
        """
        Initialize the centroids as k random samples of X.
        """
        random_idxs = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_idxs, :]

    def assign_clusters(self, X):
        """
        Assign each instance of X to the nearest centroid.
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = euclidean_distance(X, self.centroids[i])
        self.labels = np.argmin(distances, axis=1)

    def update_centroids(self, X):
        """
        Update each centroid to be the mean of the instances that are assigned to it.
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            new_centroids[i] = np.mean(X[self.labels == i, :], axis=0)
        return new_centroids

    def fit(self, X):
        """
        Run the K-means algorithm on the data X.
        """
        self.init_centroids(X)
        for _ in range(self.max_iter):
            self.assign_clusters(X)
            new_centroids = self.update_centroids(X)
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break  # Stop early if centroids didn't significantly change
            else:
                self.centroids = new_centroids

    def predict(self, X):
        """
        Predict the cluster of each instance in X based on the trained centroids.
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = euclidean_distance(X, self.centroids[i])
        return np.argmin(distances, axis=1)