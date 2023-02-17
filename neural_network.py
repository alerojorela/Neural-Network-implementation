import numpy as np
import math

from activations import *
from normalization import *

"""
This neural network uses the following arrays shapes:
W.shape = (output_units, input_units)
X.shape = (input_units, data_samples)
b.shape = (output_units, 1)
"""


class Nn:
    def __init__(self):
        """
        parameters -- python dictionary containing parameters W and b, one for each layer:
        hyper -- hyperparameters
        L -- layers number
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        """
        self.parameters = {}
        self.hyper = {}
        self.L = 0
        # convergence
        self.v = {}
        self.s = {}

    def _initialize_parameters(self):
        """
        creates a python dictionary containing your self.parameters "W1", "b1", ..., "WL", "bL":
                Wl -- weight matrix of shape (self.hyper['shape'][l], self.hyper['shape'][l-1])
                bl -- bias vector of shape (self.hyper['shape'][l], 1)
        Random initialization is used to break symmetry and make sure different hidden units can learn different things
        Resist initializing to values that are too large!
        He initialization works well for networks with ReLU activations                        
        """
        for l in range(1, self.L + 1):
            # coef = 0.01
            # coef = np.sqrt(2 / layer_dims[l - 1])
            he = (2. / self.hyper['shape'][l - 1]) ** 0.5
            self.parameters['W' + str(l)] = he * np.random.randn(self.hyper['shape'][l], self.hyper['shape'][l - 1])
            self.parameters['b' + str(l)] = np.zeros((self.hyper['shape'][l], 1))

            # initializes momentum or ADAM
            self.v["dW" + str(l)] = np.zeros(self.parameters["W" + str(l)].shape)
            self.v["db" + str(l)] = np.zeros(self.parameters["b" + str(l)].shape)
            # initializes ADAM
            self.s["dW" + str(l)] = np.zeros(self.parameters["W" + str(l)].shape)
            self.s["db" + str(l)] = np.zeros(self.parameters["b" + str(l)].shape)

    def _linear_activation_forward(self, A_prev, W, b, activation_function, D):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
    
        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        D -- array for units deactivation (DROPOUT) used for regularization
    
        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        Z = np.matmul(W, A_prev) + b
        A, backward_activation_function = activation_function(Z)
        if 'keep_units' in self.hyper and D is not None:
            A *= D
        cache = (A_prev, W, b, Z, backward_activation_function, D)
        return A, cache

    def _forward(self, X, training_time=False):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        training_time -- boolean, if True dropout is enabled

        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of _linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """
        caches = []
        A = X

        # The for loop starts at 1 because layer 0 is the input
        for l in range(1, self.L + 1):
            A_prev = A

            # DROPOUT ONLY IN TRAINING
            if training_time and 'keep_units' in self.hyper and self.hyper['keep_units'][l] != 1:
                """
                D1 = np.random.rand(*A.shape)
                D1 = (D1 < keep_prob).astype(int) / keep_prob
                A *= D1 
                """
                D = np.random.rand(self.hyper['shape'][l], 1)  # deactivate output units: will broadcast
                D = (D < self.hyper['keep_units'][l]).astype(int) / self.hyper['keep_units'][l]
            else:
                D = None

            A, cache = self._linear_activation_forward(A_prev,
                                                       self.parameters['W' + str(l)], self.parameters['b' + str(l)],
                                                       self.hyper['activation_function'][l],
                                                       D)
            caches.append(cache)

        return A, caches

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).
    
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    
        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]
        logprobs = np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))

        # loss = 1./m * np.nansum(logprobs)
        # FIXME:
        cost = -1 / m * np.sum(logprobs)
        # Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
        # cost = -1. / m * np.nansum(logprobs)

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        if self.hyper.get('λ'):
            # W layer matrix to scalar sum then, sum all layers W
            Ws = np.sum([np.sum(np.square(self.parameters[w]))
                         for w in self.parameters
                         if w[0] == 'W'])
            L2_regularization_cost = self.hyper['λ'] / 2 / m * Ws
            cost += L2_regularization_cost

        return cost

    def _linear_activation_backward(self, dA, cache):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b, Z, backward_activation_function, D = cache

        # DROPOUT always available at backward propagation
        if 'keep_units' in self.hyper and D is not None:
            # dA_prev *= D
            dA *= D

        dZ = backward_activation_function(dA, Z)

        m = A_prev.shape[1]
        dW = 1 / m * np.matmul(dZ, A_prev.T)
        if self.hyper.get('λ'):
            dW += self.hyper['λ'] / m * W

        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.matmul(W.T, dZ)

        """
        dZ3 = A3 - Y
        dW3 = 1./m * np.dot(dZ3, A2.T)
        db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
        dA2 = np.dot(W3.T, dZ3)
        """

        return dA_prev, dW, db

    def _backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
        Arguments:
        AL -- probability vector, output of the forward propagation (forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of _linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of _linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA_prev_temp = dAL
        for l in reversed(range(L)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(dA_prev_temp, current_cache)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    # region Update functions

    def _update_parameters(self, grads, learning_rate, t):
        """
        Updates self.parameters using gradient descent

        Arguments:
        grads -- python dictionary containing your gradients for each parameters:
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
        learning_rate -- the learning rate, scalar.
        t -- Adam variable, counts the number of taken steps
        """
        for l in range(1, self.L + 1):
            self.parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    def _update_parameters_with_momentum(self, grads, learning_rate, t):
        """
        Updates parameters using Momentum

        Arguments:
        grads -- python dictionary containing your gradients for each parameters:
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
        learning_rate -- the learning rate, scalar.
        t -- Adam variable, counts the number of taken steps
        """
        # beta -- the momentum hyperparameter, scalar
        beta = self.hyper['β']
        for l in range(1, self.L + 1):
            dW = "dW" + str(l)
            db = "db" + str(l)
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            self.v[dW] = beta * self.v[dW] + (1 - beta) * grads[dW]
            self.v[db] = beta * self.v[db] + (1 - beta) * grads[db]
            self.parameters["W" + str(l)] -= learning_rate * self.v[dW]
            self.parameters["b" + str(l)] -= learning_rate * self.v[db]

    def _update_parameters_with_adam(self, grads, learning_rate, t):
        """
        Updates parameters using Adam

        Arguments:
        grads -- python dictionary containing your gradients for each parameters:
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
        learning_rate -- the learning rate, scalar.
        t -- Adam variable, counts the number of taken steps
        """
        # try values: beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8
        beta1 = self.hyper['β1']
        beta2 = self.hyper['β2']
        epsilon = 1e-8

        v_corrected = {}  # Initializing first moment estimate, python dictionary
        s_corrected = {}  # Initializing second moment estimate, python dictionary

        # Perform Adam update on all parameters
        for l in range(1, self.L + 1):
            dW = "dW" + str(l)
            db = "db" + str(l)
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            self.v[dW] = beta1 * self.v[dW] + (1 - beta1) * grads[dW]
            self.v[db] = beta1 * self.v[db] + (1 - beta1) * grads[db]
            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected[dW] = self.v[dW] / (1 - beta1 ** t)
            v_corrected[db] = self.v[db] / (1 - beta1 ** t)

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            self.s[dW] = beta2 * self.s[dW] + (1 - beta2) * grads[dW] ** 2
            self.s[db] = beta2 * self.s[db] + (1 - beta2) * grads[db] ** 2
            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected[dW] = self.s[dW] / (1 - beta2 ** t)
            s_corrected[db] = self.s[db] / (1 - beta2 ** t)

            self.parameters["W" + str(l)] -= learning_rate * v_corrected[dW] / (
                    np.sqrt(s_corrected[dW]) + epsilon)
            self.parameters["b" + str(l)] -= learning_rate * v_corrected[db] / (
                    np.sqrt(s_corrected[db]) + epsilon)

    # endregion

    # region Interface
    def train(self, X, Y, hyperparameters):
        """
        Generator function
        Trains a L-layer neural network for classification

        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- boolean vector of shape (1, number of examples)
        hyperparameters: dictionary. () indicates optional keys
            shape
            activation_function
            (λ)
            (keep_units)
            learning_rate
            epochs
            (β)
            (β1 β2)
        yields:
        cost
        """
        self.parameters = {}
        self.hyper = hyperparameters
        # L - 1 layers
        self.L = len(self.hyper['shape']) - 1  # number of layers in the neural network
        # convergence
        self.v = {}
        self.s = {}

        # infer update function by hyperparameters variables
        if self.hyper.get('β'):
            print('Momentum optimization')
            update_function = self._update_parameters_with_momentum
        elif self.hyper.get('β1') and self.hyper.get('β2'):
            print('Adam optimization')
            update_function = self._update_parameters_with_adam
        else:
            update_function = self._update_parameters

        self._initialize_parameters()

        costs = []  # keep track of cost
        # Main loop (forward and backward steps)
        for i in range(0, self.hyper['epochs']):
            # Forward propagation: loop 1...L
            AL, caches = self._forward(X, training_time=True)
            cost = self.compute_cost(AL, Y)
            # Backward propagation: loop L...1
            grads = self._backward(AL, Y, caches)

            learning_rate = self.hyper['learning_rate']
            # decaying α functions:
            # learning_rate = learning_rate0 / (1 + decay_rate * epoch_num)
            # learning_rate = learning_rate0 / (1 + decay_rate * math.floor(epoch_num / time_interval))
            update_function(grads, learning_rate, i + 1)

            # rel_tol = value 	Optional. The relative tolerance. It is the maximum allowed difference between value a and b. Default value is 1e-09
            if costs and math.isclose(costs[-1], cost, rel_tol=1e-07):
                # if costs and math.isclose(costs[-1], cost):
                print('convergence ', costs[-1], cost)
                return cost

            costs.append(cost)
            yield cost

    def predict(self, X):
        """
        This function is used to predict the results of a n-layer neural network.

        Arguments:
        X -- input data of size (n_x, m)
        Returns
        predictions -- vector of predictions of our model
        """
        AL, caches = self._forward(X)
        return AL > 0.5

    def accuracy(self, X, Y):
        """
        Accuracy function: prediction vs annotated value

        Arguments:
        X -- input data set
        Y -- supervised output data set

        Returns:
        p -- predictions for the given dataset X
        acc -- mean of correct predictions
        """
        AL = self.predict(X)
        acc = np.mean((AL[0, :] == Y[0, :]))
        return AL, acc
    # endregion
