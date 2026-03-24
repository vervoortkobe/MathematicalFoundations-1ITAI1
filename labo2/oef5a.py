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

# Opdracht 5a
print("Opdracht 5a: Dot products\n")

# Paar 1
u1, v1 = np.array([1, 2, 3]), np.array([4, 5, 6])
dot1 = u1 @ v1
print(f"1. u = {u1}, v = {v1}")
print(f"   u · v = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = {dot1}")
print()

# Paar 2
u2, v2 = np.array([1, 0, 0]), np.array([0, 1, 0])
dot2 = u2 @ v2
print(f"2. u = {u2}, v = {v2}")
print(f"   u · v = 1×0 + 0×1 + 0×0 = {dot2} (loodrecht!)")
print()

# Paar 3
u3, v3 = np.array([2, 3]), np.array([-3, 2])
dot3 = u3 @ v3
print(f"3. u = {u3}, v = {v3}")
print(f"   u · v = 2×(-3) + 3×2 = -6 + 6 = {dot3} (ook loodrecht!)")