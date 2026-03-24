# Voer deze cel eerst uit om alle benodigde libraries te importeren

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

np.set_printoptions(precision=3, suppress=True)

# Laad MNIST
print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

class Layer:
    def __init__(self, input_size, output_size, activation='none'):
        self.activation = activation
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        z = x @ self.W + self.b
        if self.activation == 'relu':
            return self.relu(z)
        elif self.activation == 'softmax':
            return self.softmax(z)
        else:
            return z

    def __repr__(self):
        return f"Layer(input_size={self.W.shape[0]}, output_size={self.W.shape[1]})"

network = [
    Layer(784, 256, activation='relu'),
    Layer(256, 128, activation='relu'),
    Layer(128, 10, activation='softmax')
]

def forward_network(X, layers):
    out = X
    for layer in layers:
        out = layer.forward(out)
    return out

batch = X[:10]
outputs = forward_network(batch, network)

print("Outputvorm:", outputs.shape)
print("Voorspelde labels:", np.argmax(outputs, axis=1))

for i, layer in enumerate(network, 1):
    print(f"Laag {i}: {layer}")
