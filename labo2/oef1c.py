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

# Opdracht 1c
print("Opdracht 1c: Willekeurige vector en statistieken\n")

np.random.seed(42)  # Voor reproduceerbaarheid
random_vec = np.random.randint(1, 101, size=10)

print(f"Random vector: {random_vec}")
print(f"Minimum: {random_vec.min()}")
print(f"Maximum: {random_vec.max()}")
print(f"Gemiddelde: {random_vec.mean():.2f}")
print(f"Som: {random_vec.sum()}")