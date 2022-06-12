import numpy as np
from collections import Counter

# TODO: shift this function into a utility class or a seperate program
def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, test_x):
        predicted_labels = [self._predict(x) for x in test_x]
        return np.array(predicted_labels)

    def _predict(self, x):
        #compute distances
        distances = [euclidian_distance(x, x_train) for x_train in self.x_train]
        #get the k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        #majority vote to get most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

