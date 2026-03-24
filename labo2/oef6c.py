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

# Opdracht 6c
print("Opdracht 6c: Random neuron op MNIST\n")

np.random.seed(42)
random_weights = np.random.uniform(-0.1, 0.1, size=784)
bias = 0

print("Output van random neuron op 10 MNIST afbeeldingen:\n")
for i in range(10):
    img = X[i]
    output = neuron(img, random_weights, bias)
    print(f"  Afbeelding {i} (label {y[i]}): output = {output:.2f}")

print("\nDe outputs zijn willekeurig en hebben geen relatie met de labels.")
print("Het netwerk moet eerst getraind worden om zinvolle outputs te geven.")