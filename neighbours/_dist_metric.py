import numpy as np


# I mirrored the class implementation from scikit-learn's code but I implemented the metrics from their formulas myself.

class DistMetric:
    def __init__(self):
        self.p = 2
        self.size = 1

    @classmethod
    def get_metric(cls, metric, **kwargs):
        if isinstance(metric, DistMetric):
            return metric

        # Map the metric string ID to the metric class
        if isinstance(metric, type) and issubclass(metric, DistMetric):
            pass
        else:
            try:
                metric = METRIC_MAPPING[metric]
            except KeyError:
                raise ValueError("Unrecognized metric '%s'" % metric)

        # In Minkowski special cases, return more efficient methods
        if metric is MinkowskiDist:
            p = kwargs.pop('p', 2)
            if p == 1:
                return ManhattanDist(**kwargs)
            elif p == 2:
                return EuclideanDist(**kwargs)
            elif np.isinf(p):
                return ChebyshevDist(**kwargs)
            else:
                return MinkowskiDist(p, **kwargs)
        else:
            return metric(**kwargs)

    def dist(self, x1, x2):
        """Compute the distance between vectors x1 and x2
        This should be overridden in a base class.
        """
        return -999


class EuclideanDist(DistMetric):
    def __init__(self):
        self.p = 2

    def dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))


class ManhattanDist(DistMetric):
    def __init__(self):
        self.p = 1

    def dist(self, x1, x2):
        return np.sum(np.abs(x1 - x2), axis=1)


class ChebyshevDist(DistMetric):
    def __init__(self):
        self.p = np.inf

    def dist(self, x1, x2):
        return np.max(np.abs(x1 - x2))


class MinkowskiDist(DistMetric):
    def __init__(self, p):
        if p < 1:
            raise ValueError("p must be greater than 1")
        elif np.isinf(p):
            raise ValueError("MinkowskiDist requires finite p. "
                             "For p=inf, use ChebyshevDist.")
        self.p = p

    def dist(self, x1, x2):
        return np.sum(np.abs(x1 - x2) ** self.p, axis=1) ** (1 / self.p)


class HammingDist(DistMetric):
    def dist(self, x1, x2):
        assert len(x1) == len(x2)

        return np.sum(x1 != x2, axis=1) / len(x1)


class CanberraDist(DistMetric):
    def dist(self, x1, x2):
        num = np.abs(x1 - x2)
        denom = np.abs(x1) + np.abs(x2)

        return np.sum(num / denom, axis=1)


######################################################################
# metric mappings
#  These map from metric id strings to class names

METRIC_MAPPING = {'euclidean': EuclideanDist,
                  'l2': EuclideanDist,
                  'minkowski': MinkowskiDist,
                  'p': MinkowskiDist,
                  'manhattan': ManhattanDist,
                  'cityblock': ManhattanDist,
                  'l1': ManhattanDist,
                  'chebyshev': ChebyshevDist,
                  'infinity': ChebyshevDist,
                  'hamming': HammingDist,
                  'canberra': CanberraDist}
