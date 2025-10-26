import numpy as np
from collections import Counter

class KNN():
    def __init__(self, k: int = 3):
        self.k = k

    def euclidean_distance(self, x1, x2):
        distance = np.sqrt(np.sum(x1 - x2)**2)
        return distance 
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        predict_method = np.array(predicted_labels)
       
        return predict_method

    
    def _predict(self, x):
        # Compute distances
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        #  get the k nearest labels
        k_indices = np.argsort(distances)[:self.k]
        
        # return the most common class label
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]