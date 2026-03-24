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

# Opdracht 6a
def neuron(inputs, weights, bias):
    """Bereken de output van een enkel neuron."""
    return inputs @ weights + bias

inputs = np.array([0.5, 0.3, 0.2])
weights = np.array([0.4, 0.6, -0.2])
bias = 0.1

output = neuron(inputs, weights, bias)

print("Opdracht 6a: Neuron functie\n")
print(f"Inputs: {inputs}")
print(f"Weights: {weights}")
print(f"Bias: {bias}")
print()
print(f"Berekening: 0.5×0.4 + 0.3×0.6 + 0.2×(-0.2) + 0.1")
print(f"          = 0.2 + 0.18 - 0.04 + 0.1")
print(f"          = {output}")