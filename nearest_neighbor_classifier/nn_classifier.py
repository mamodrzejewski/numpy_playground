import numpy as np


class NearestNeighborClassifier:
    """
    Very simple NN classifier written in numpy.
    """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Remembers all the training data.

        X is (N x D) training data, each row is an example.
        y is 1 dimension of size N and contains the labels.
        """
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """
        Finds the nearest training image to the i'th test image.
        Uses L1 distance.

        X is (N x D) examples for prediction.
        """
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows and compute L1 distance
        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)
            # now we can get the label of the nearest example
            Ypred[i] = self.ytr[min_index]

        return Ypred


if __name__ == "__main__":
    # create some artificial data
    data = [np.random.random((1, 3)) + offset for offset in range(3)]
    X = np.vstack(data)
    # create artificial labels for data
    y = np.arange(0, 3).reshape(3, 1)

    # create and train a nn classifier
    nn = NearestNeighborClassifier()
    nn.train(X, y)

    # predict for some artificial test examples
    Xtest = np.array(([3, 3, 3], [0, 0, 0]))
    print(nn.predict(Xtest))
