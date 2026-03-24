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

# Opdracht 4a
print("Opdracht 4a: Norm berekenen\n")

v1 = np.array([3, 4])
v2 = np.array([1, 1, 1, 1])
v3 = np.array([5, 0, 0, 0, 0])

print(f"v1 = {v1}")
print(f"  ||v1|| = √(3² + 4²) = √(9 + 16) = √25 = 5")
print(f"  NumPy: {np.linalg.norm(v1)}")
print()

print(f"v2 = {v2}")
print(f"  ||v2|| = √(1² + 1² + 1² + 1²) = √4 = 2")
print(f"  NumPy: {np.linalg.norm(v2)}")
print()

print(f"v3 = {v3}")
print(f"  ||v3|| = √(5² + 0² + 0² + 0² + 0²) = √25 = 5")
print(f"  NumPy: {np.linalg.norm(v3)}")