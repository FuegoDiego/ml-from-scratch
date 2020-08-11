# Implementing Machine Learning Algorithms from Scratch
Repository for implementing machine learning methods from scratch.

The following algorithms have been implemented:
## k-Nearest Neighbours
This method was implemented with the option to change the distance metric. The following metrics are available to use:
* Euclidean Distance
* Minkowski Distance
* Manhattan Distance
* Chebyshev Distance
* Hamming Distance
* Canberra Distance

The option to add weight voting is available as well. The weight option 'distance' weighs each class label by the inverse of the distance to the training data.

The algorithm was tested by recreating [this](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py) sample usage from scikit-learn. The images below were produced using this implementation of the algorithm and match the ones achieved by the scikit-learn implementation. Both runs used the euclidean distance and only the second one (second image) used weight voting.
![Euclidean Distance Without Weight Voting](/knn/images/uniform-distance.jpeg)
![Euclidean Distance With Weight Voting](/knn/images/inverse-distance.jpeg)
