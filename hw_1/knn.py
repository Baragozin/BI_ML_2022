import numpy as np
from scipy import stats


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.y_train)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        shape = (np.shape(X)[0], np.shape(self.X_train)[0])
        dist_mat = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                dist_mat[i, j] += np.sum(abs(X[i] - self.X_train[j]))
        return dist_mat


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        shape = (np.shape(X)[0], np.shape(self.X_train)[0])
        dist_mat = np.zeros(shape)

        for i in range(shape[0]):
            dist_mat[i] += np.sum(abs(X[i] - self.X_train), axis=1)
        
        return dist_mat


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        rows_test, cols_test = X.shape
        rows_train = self.X_train.shape[0]
        return abs(X.reshape(rows_test, 1, cols_test) - self.X_train).sum(2).reshape(rows_test, rows_train)


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)

        for i in range(distances.shape[0]):
            k_indices = np.argsort(distances[i])[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            predicted = np.argmax(np.bincount(k_nearest_labels))
            prediction[i] = predicted
        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        for i in range(distances.shape[0]):
            k_indices = np.argsort(distances[i])[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            predicted = np.argmax(np.bincount(k_nearest_labels))
            prediction[i] = predicted
        return prediction
