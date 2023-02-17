import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
import scipy.io


def load(fn, visualize=False):
    train_X, train_Y, test_X, test_Y = fn(visualize)

    print("\nDATA INFO: ")
    print("train_X shape: " + str(train_X.shape))
    print("train_Y shape: " + str(train_Y.shape))
    print("test_X shape: " + str(test_X.shape))
    print("test_Y shape: " + str(test_Y.shape))

    assert train_X.shape[0] == train_Y.shape[0]
    assert test_X.shape[0] == test_Y.shape[0]
    assert train_X.shape[1] == test_X.shape[1]
    assert train_Y.shape[1] == test_Y.shape[1]
    print(f"\nNumber of input features: {train_X.shape[1]}")
    print(f"Number of output features: {train_Y.shape[1]}")
    print(f"Number of training samples: {train_X.shape[0]}")
    print(f"Number of test samples: {test_X.shape[0]}")

    return train_X.T, train_Y.T, test_X.T, test_Y.T

def halfs_dataset(visualize=False):
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X']
    train_Y = data['y']
    test_X = data['Xval']
    test_Y = data['yval']

    if visualize:
        # Visualize the data
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
        plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y, s=40, cmap=plt.cm.Spectral)
        plt.plot()
        plt.show()

    return train_X, train_Y, test_X, test_Y


def concentric_dataset(visualize=False):
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    train_Y = train_Y.reshape((train_Y.shape[0], 1))
    test_Y = test_Y.reshape((test_Y.shape[0], 1))

    if visualize:
        # Visualize the data
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
        plt.plot()
        plt.show()

    return train_X, train_Y, test_X, test_Y


def petals_dataset(visualize):
    np.random.seed(1)

    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    train_X, test_X = train_test_split(X, test_size=0.2, random_state=1, shuffle=True)
    train_Y, test_Y = train_test_split(Y, test_size=0.2, random_state=1, shuffle=True)

    if visualize:
        # Visualize the data
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y)
        plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y)
        plt.plot()
        plt.show()

    return train_X, train_Y, test_X, test_Y


def spiral_dataset(visualize=False):
    randomness = 1
    np.random.seed(1)

    m = 50
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 2  # maximum ray of the flower

    for j in range(2):

        ix = range(N * j, N * (j + 1))
        if j == 0:
            t = np.linspace(j, 4 * 3.1415 * (j + 1), N)  # + np.random.randn(N)*randomness # theta
            r = 0.3 * np.square(t) + np.random.randn(N) * randomness  # radius
        if j == 1:
            t = np.linspace(j, 2 * 3.1415 * (j + 1), N)  # + np.random.randn(N)*randomness # theta
            r = 0.2 * np.square(t) + np.random.randn(N) * randomness  # radius

        X[ix] = np.c_[r * np.cos(t), r * np.sin(t)]
        Y[ix] = j

    train_X, test_X = train_test_split(X, test_size=0.2, random_state=1, shuffle=True)
    train_Y, test_Y = train_test_split(Y, test_size=0.2, random_state=1, shuffle=True)

    if visualize:
        # Visualize the data
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y)
        plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y)
        plt.plot()
        plt.show()

    return train_X, train_Y, test_X, test_Y


def load_image_dataset():
    # cat images classification
    import h5py
    # samples, dim x dim, RGB layers
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_Y = train_set_y_orig.reshape((-1, 1))
    test_Y = test_set_y_orig.reshape((-1, 1))

    train_X = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
    test_X = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)
    # normalization
    train_X = train_X / 255
    test_X = test_X / 255

    return train_X.T, train_Y.T, test_X.T, test_Y.T
    # return train_set_x, train_set_y, test_set_x, test_set_y, classes


def forms(N=200):
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
