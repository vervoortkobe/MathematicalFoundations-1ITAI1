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

# Opdracht 3b
print("Opdracht 3b: Commutativiteit en associativiteit\n")

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
w = np.array([7, 8, 9])

# Commutativiteit: u + v = v + u
commutatief = np.allclose(u + v, v + u)
print(f"Commutativiteit (u + v = v + u): {commutatief}")
print(f"  u + v = {u + v}")
print(f"  v + u = {v + u}")
print()

# Associativiteit: (u + v) + w = u + (v + w)
associatief = np.allclose((u + v) + w, u + (v + w))
print(f"Associativiteit ((u + v) + w = u + (v + w)): {associatief}")
print(f"  (u + v) + w = {(u + v) + w}")
print(f"  u + (v + w) = {u + (v + w)}")