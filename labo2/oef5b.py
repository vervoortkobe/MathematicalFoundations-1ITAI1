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

# Opdracht 5b
print("Opdracht 5b: Commutativiteit van dot product\n")

u = np.array([1, 2, 3, 4, 5])
v = np.array([5, 4, 3, 2, 1])

dot_uv = u @ v
dot_vu = v @ u

print(f"u = {u}")
print(f"v = {v}")
print()
print(f"u · v = {dot_uv}")
print(f"v · u = {dot_vu}")
print(f"\nCommutatief: {dot_uv == dot_vu}")