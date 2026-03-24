# Imports en data laden
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

def cosine_similarity(u, v):
    """Bereken de cosine similarity tussen twee vectoren."""
    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Opdracht 6b
def relu(x):
    """ReLU activatiefunctie."""
    return np.maximum(0, x)

def neuron_relu(inputs, weights, bias):
    """Neuron met ReLU activatie."""
    weighted_sum = inputs @ weights + bias
    return relu(weighted_sum)

print("Opdracht 6b: Neuron met ReLU\n")

inputs = np.array([0.5, 0.3, 0.2])

# Test met verschillende gewichten
test_cases = [
    (np.array([0.4, 0.6, -0.2]), 0.1, "positieve output"),
    (np.array([-0.4, -0.6, -0.2]), 0.1, "negatieve gewogen som"),
    (np.array([0.1, 0.1, 0.1]), -0.5, "negatieve door bias"),
]

for weights, bias, beschrijving in test_cases:
    raw = inputs @ weights + bias
    activated = neuron_relu(inputs, weights, bias)
    print(f"{beschrijving}:")
    print(f"  Weights: {weights}, Bias: {bias}")
    print(f"  Gewogen som: {raw:.4f}")
    print(f"  Na ReLU: {activated:.4f}")
    print()