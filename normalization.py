import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def z_score(x):
    scaler_linear = StandardScaler()
    # Compute the mean and standard deviation of the training set then transform it
    scaled = scaler_linear.fit_transform(x)
    return scaled


def normalize_rows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
