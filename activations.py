import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    return np.reciprocal(1 + np.exp(-Z)), sigmoid_backward


def relu(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    return A, relu_backward


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


# def sigmoid_derivative(x):
def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache

    s, _ = sigmoid(Z)
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)
    return dZ


def softmax(x):
    """Calculates the softmax for each row of the input x.
    Your code should work for a row vector and also for matrices of shape (m,n).
    Argument:
    x -- A numpy matrix of shape (m,n)
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    # You can think of softmax as a normalizing function used when your algorithm needs to classify two or more classes
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)  # preserves rows, collapses column to one
    s = x_exp / x_sum
    return s
