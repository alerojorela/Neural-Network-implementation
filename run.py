""" SETUP
pip install numpy scikit-learn matplotlib scipy h5py
"""
import numpy as np
import matplotlib.pyplot as plt

import neural_network
from activations import relu, sigmoid
import data_load

# load data. ALTERNATIVES:
# train_X, train_Y, test_X, test_Y = data_load.load(data_load.halfs_dataset, visualize=True)
# train_X, train_Y, test_X, test_Y = data_load.load(data_load.concentric_dataset, visualize=True)
train_X, train_Y, test_X, test_Y = data_load.load(data_load.petals_dataset, visualize=True)
# FIXME: INACCURATE MODEL:
# train_X, train_Y, test_X, test_Y = data_load.load(data_load.spiral_dataset, visualize=True)
# FIXME: ERROR: images
# train_X, train_Y, test_X, test_Y = data_load.load_image_dataset()

# input_units
input_units = train_X.shape[0]
output_units = train_Y.shape[0]

"""
Parameters vs Hyperparameters
    Parameters
        W b
    Hyperparameters
        design
            shape -- list containing the input size and each layer size, of length (number of layers + 1).
                L layers
                    hidden layers
                n[1], ..., n[L]
                    units per layer        
            g activation_function
        regularization
            𝜆 L2 regularization makes your decision boundary smoother. 
            If 𝜆 is too large, it is also possible to "oversmooth", resulting in a model with high bias.
            keep_units

        α learning_rate -- learning rate of the gradient descent update rule
            learning decay rate
        epochs -- number of iterations of the optimization loop
        optimizer
            Momentum
                β
            Adam
                β1 -- Exponential decay hyperparameter for the first moment estimates
                β2 -- Exponential decay hyperparameter for the second moment estimates
                ε -- hyperparameter preventing division by zero in Adam updates       
"""
hyperparameters = {
    'shape': [input_units, 10, 5, output_units],
    # 'shape': [input_units, 64, 16, output_units],
    'activation_function': [None, relu, relu, sigmoid],
    # regularization:
    # 'λ': 4,
    'λ': 1,
    # 'keep_units': [None, 0.95, 0.95, 1],
    # 'keep_units': [None, 1, 1, 1],
    # convergence:
    'learning_rate': 0.0075,
    'epochs': 15000,
    # 'β': 0.9,
    'β1': 0.9, 'β2': 0.999,  # ADAM
}

nn = neural_network.Nn()

costs = []
generator = nn.train(train_X, train_Y, hyperparameters)
for epoch, cost in enumerate(generator):
    # Print the cost every 100 iterations
    if epoch % 100 == 0 or epoch == hyperparameters['epochs'] - 1:
        print("Cost after iteration {}: {}".format(epoch, cost))
        costs.append(cost)
        # print(epoch, cost)

p1, acc1 = nn.accuracy(train_X, train_Y)
p2, acc2 = nn.accuracy(test_X, test_Y)
print(f"""
HYPERPARAMETERS
---------------
convergence:
    epochs: {hyperparameters['epochs']}
    learning_rate: {hyperparameters.get('learning_rate')}
    β: {hyperparameters.get('β')}
    ADAM:
        β1: {hyperparameters.get('β1')}
        β2: {hyperparameters.get('β2')}
regularization:
    λ: {hyperparameters.get('λ')}
    keep_units: {hyperparameters.get('keep_units')}

RESULTS
-------
Convergence epochs: {epoch + 1}
Final cost: {costs[-1]:.3g}
TRAIN  Accuracy: {acc1:.3g}
TEST Accuracy: {acc2:.3g}
Baseline: ?
Bias: baseline to {1 - acc1:.3g}
Variance: {acc1 - acc2:.3g}

""")


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

    plt.show()


# Classification boundaries
plot_decision_boundary(lambda x: nn.predict(x.T), train_X, train_Y)

# Cost function evolution
r = 100 * np.arange(len(costs))
plt.plot(r, costs)
plt.title('Cost function evolution')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()
