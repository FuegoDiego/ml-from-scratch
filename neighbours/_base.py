import numpy as np

"""I mirrored the class implementations from the scikit-learn source code."""


def _check_weights(weights):
    """Check that weights are valid"""
    if weights in (None, 'uniform', 'distance'):
        return weights
    else:
        raise ValueError("weights is not recognized: must be one of 'uniform' or 'distance'")


def _get_weights(dist, weights):
    """Get the weights from an array of distance values and the type of weight calculation

    Parameters
    ----------
    dist : numpy.ndarray
        The input distances
    weights : str
        {'uniform', 'distance'}

    Returns
    -------
    weights_arr : array of the same shape as 'dist'
        if weights == 'uniform', then returns None
    """

    if weights in (None, 'uniform'):
        return None
    elif weights == 'distance':
        dist = dist.reshape(len(dist), 1)
        # if a point has zero distance from one or more training points, those training points will have a weight of 1
        # and all other training points will have a weight of 0
        with np.errstate(divide='ignore'):
            dist = 1. / dist
        is_inf_dist = np.isinf(dist)
        is_inf_row = np.any(is_inf_dist, axis=1)
        dist[is_inf_row] = is_inf_dist[is_inf_row]

        return dist.flatten()
    else:
        raise ValueError("weights is not recognized: must be one of 'uniform' or 'distance'")
