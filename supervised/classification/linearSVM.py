import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class LinearSVM:
    def __init__(self, learning_rate=0.001, _lambda=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self._lambda = _lambda
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        _, n_features = X.shape

        _y = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for index, x_i in enumerate(X):
                condition = _y[index] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self._lambda * self.weights)
                else: 
                    self.weights -= self.learning_rate * (2 * self._lambda * self.weights - np.dot(x_i, _y[index]))
                    self.bias -= self.learning_rate * _y[index]

    def predict(self, X):
        approximation = np.dot(X, self.weights) - self.bias
        return np.sign(approximation)

def main():
    # Load data and create train-test split
    iris_data = load_breast_cancer()
    train_X, test_X, train_y, test_y = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=2)

    # Initalise, train the SVM and predict
    svm = LinearSVM()
    svm.fit(train_X, train_y)
    predictions = svm.predict(test_X)

    # Use vectorisation to replace all negatives to 0
    # vectorisation is GIGA efficient
    replace_negatives = np.vectorize(lambda x:0 if x < 0 else x)
    regularized_predictions = replace_negatives(predictions)
    correct_predictions = 0


    # Compare predictions with target values by counting same values
    for index, prediction in enumerate(regularized_predictions):
        if prediction == test_y[index]:
            correct_predictions += 1

    print(regularized_predictions)
    print(test_y)
    
    print(f'Number of correct predictions: {correct_predictions} \n Total number of predictions: {len(test_y)} \n Percentage of correct predictions: {correct_predictions/len(test_y)}')

if __name__ == "__main__":
    main()