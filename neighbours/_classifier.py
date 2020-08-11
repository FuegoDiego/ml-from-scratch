import numpy as np
from collections import Counter
from sklearn.utils.extmath import weighted_mode

from ._base import _check_weights, _get_weights
from ._dist_metric import DistMetric

"""I mirrored the class implementations from the scikit-learn source code, but I implemented the KNN myself"""


class KNeighboursClassifier:
    def __init__(self, n_neighbours=5, weights='uniform', p=2, metric='minkowski', metric_params=None):
        self.n_neighbours = n_neighbours
        self.weights = _check_weights(weights)
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.classes_ = None  # this is initialized in the fit method
        self.X_train_ = None  # this is initialized in the fit method

    def fit(self, X, y):
        if self.metric == 'minkowski':
            if self.p < 1:
                raise ValueError("The value of p must be greater than 1 for the 'minkowski' distance")
            elif self.p == 1:
                self.metric = 'manhattan'
            elif self.p == 2:
                self.metric = 'euclidean'
            elif self.p == np.inf:
                self.metric = 'chebyshev'
            else:
                self.metric_params['p'] = self.p
        if self.n_neighbours < 0:
            raise ValueError("The number of neighbours must be greater than 1")

        self.classes_ = np.array(y)
        self.X_train_ = np.array(X)

        return self

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        k = self.n_neighbours

        if self.metric_params is None:
            dist_metric = DistMetric.get_metric(self.metric)
        else:
            dist_metric = DistMetric.get_metric(self.metric, *self.metric_params)
        dist = dist_metric.dist(self.X_train_, x)

        weights = _get_weights(dist, self.weights)

        if weights is None:
            # zip together in order to sort then unzip to get back the sorted lists
            zipped = sorted(zip(dist, self.classes_))
            dist, class_labels = zip(*zipped)

            # get the labels of the k closest points to x
            class_labels = np.array(class_labels)[:k]

            # get the most common label
            predict_label = Counter(class_labels).most_common(1)[0][0]

            return predict_label
        else:
            zipped = sorted(zip(dist, self.classes_, weights))
            dist, class_labels, weights = zip(*zipped)

            class_labels = np.array(class_labels)[:k]
            weights = np.array(weights)[:k]

            predict_label = weighted_mode(class_labels, weights)[0][0]

            return predict_label
