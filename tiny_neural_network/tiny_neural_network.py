import numpy as np


def sigmoid(x, derivative: bool = False):
    if(derivative is True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


np.random.seed(1)

# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T


# 2 layer network
def nn(X: np.array, y: np.array):
    # one layer: map 3 inputs to prediction
    syn0 = 2 * np.random.random((3, 1)) - 1
    print(syn0)
    for i in range(50000):
        # forward pass
        layer0 = X
        layer1 = sigmoid(np.dot(layer0, syn0))
        # compute error
        layer1_error = y - layer1
        # compute gradient
        dz = layer1_error * sigmoid(layer1, derivative=True)
        dW = layer0.T.dot(dz)
        syn0 += dW
    print("Output after training:")
    print(layer1)


nn(X, y)


# input dataset
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# output dataset
y = np.array([[0, 1, 1, 0]]).T
# 3 layer network


def nn_with_hidden_layer(X: np.array, y: np.array):
    # input layer: 3 inputs, 4 outputs
    # hidden layer: 4 inputs, 1 output
    # output layer

    # init weights
    syn0 = 2 * np.random.random((3, 4)) - 1
    syn1 = 2 * np.random.random((4, 1)) - 1

    for i in range(50000):
        # forward pass
        layer0 = X
        layer1 = sigmoid(np.dot(layer0, syn0))
        layer2 = sigmoid(np.dot(layer1, syn1))

        # compute error
        layer2_error = y - layer2
        if(i % 10000 == 0):
            print(f"Error:\n {np.mean(np.abs(layer2_error))}")

        # compute gradients
        layer2_delta = layer2_error * sigmoid(layer2, derivative=True)

        layer1_error = layer2_delta.dot(syn1.T)
        layer1_delta = layer1_error * sigmoid(layer1, derivative=True)

        # update weights
        syn1 += layer1.T.dot(layer2_delta)
        syn0 += layer0.T.dot(layer1_delta)
    print("Output after training:")
    print(layer2)


nn_with_hidden_layer(X, y)
