import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

N_SAMPLES = 2000
TEST_SIZE = 0.3

x, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

print(x.shape, y.shape)


def make_plot(x, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel="x1", ylabel="x2")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(right=0.8)
    if (XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1, camp=plt.cm.Spectral)
        plt.contourf(XX, YY, preds.reshape(XX.shape), levels=[.5], camp="Greys", vmin=0, vmax=.6)
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='none')
    plt.savefig('dataset.svg')


# make_plot(x, y, "Visualization")
# plt.show()

class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.randn(n_neurons)
        self.activation = activation
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)

    def _apply_activation(self, r):
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        elif self.activation == 'tanh':
            return 1 - r ** 2
        elif self.activation == 'sigmoid':
            return r * (1 - r)

        return r


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layers(self, layer):
        self.layers.append(layer)

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.activate(x)
        return x

    def backPropagation(self, x, y, learning_rate):
        output = self.feed_forward(x)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer == self.layers[-1]:
                layer.error = y - output
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        for i in range(len(self.layers)):
            layer = self.layers[i]

            out_last = np.atleast_2d(x if i == 0 else self.layers[i - 1].last_activation)

            layer.weights += layer.delta * learning_rate * out_last.T


nn = NeuralNetwork()

nn.add_layers(Layer(2, 25, 'sigmoid'))
nn.add_layers(Layer(25, 50, 'sigmoid'))
nn.add_layers(Layer(50, 25, 'sigmoid'))
nn.add_layers(Layer(25, 2, 'sigmoid'))
