from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt

class KNearestNeighbor:
    def __init__(self, x, y, k=3):
        self.X = x
        self.Y = y
        self.k = k

    @staticmethod
    def euclidean_distance(a1, a2):
        return np.sqrt(np.sum((a1 - a2)**2))

    @staticmethod
    def manhattan_distance(a1, a2):
        return np.sum(np.abs(a1 - a2))

    """
    params:
        x (array): A data point to classify
    returns:
        prediction (int): prediction of x 
    """
    def predict(self, x):
        neighbors = {}
        # Each row
        for idx, i in enumerate(self.X):
            distance = 0
            # Each column of row
            for j_idx, j in enumerate(i):
                distance += KNearestNeighbor.euclidean_distance(j, x[j_idx])
            if len(neighbors) < self.k:
                neighbors[idx] = distance
            if distance < max(neighbors.keys()):
                del neighbors[max(neighbors, key=neighbors.get)]
                neighbors[idx] = distance
        prediction = Counter(self.Y[i] for i in neighbors.keys())
        return prediction.most_common()[0][0]

    """
    params:
        x (array of arrays): A set of data points to classify
    returns:
        predictions (array of classes): Prediction of the data points of x 
    """
    def predict_all(self, x):
        return [self.predict(i) for i in x]
def main():
    data = load_iris()
    # plt.plot(data.data, data.target)
    # plt.show()
    train_X, test_X, train_Y, test_Y = train_test_split(data.data, data.target, test_size=0.2, random_state=2)
    # plt.plot(train_X, train_Y)
    # plt.show()
    # print(train_X, train_Y)
    # print(f"TEST Y: {test_X}")
    """
    multiple_predictions = test_X[3:6]
    for idx, i in enumerate(multiple_predictions):
        print(f"Data Point to predict {i}, actual class: {test_Y[idx+3]}")
    # print(test_Y)
    knn = KNearestNeighbor(train_X, train_Y)
    # print(knn.predict(prediction))
    prediction_results = knn.predict_all(multiple_predictions)
    for idx, i in enumerate(prediction_results):
        print(f"Data point {multiple_predictions[idx]} is predicted to be {i}")
    # print(train_X[0], train_Y[0], test_X[0], test_Y[0])
    """
    knn = KNearestNeighbor(train_X, train_Y)
    prediction_results = knn.predict_all(test_X)

    sk_knn = KNeighborsClassifier(n_neighbors=3)
    sk_knn.fit(train_X, train_Y)
    sk_predictions = sk_knn.predict(test_X)

    for idx, i in enumerate(prediction_results):
        print(f"Data point {test_X[idx]} is predicted to be {i}, sk_predict: {sk_predictions[idx]}, actual class: {test_Y[idx]}")

if __name__ == '__main__':
    main()