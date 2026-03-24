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

# Opdracht 3a
print("Opdracht 3a: Basisoperaties\n")

u = np.array([1, 2, 3, 4])
v = np.array([5, 6, 7, 8])

print(f"u = {u}")
print(f"v = {v}")
print()

print(f"u + v = {u + v}")
print(f"  Controle: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12] ✓")
print()

print(f"u - v = {u - v}")
print(f"  Controle: [1-5, 2-6, 3-7, 4-8] = [-4, -4, -4, -4] ✓")
print()

print(f"3 * u = {3 * u}")
print(f"  Controle: [3×1, 3×2, 3×3, 3×4] = [3, 6, 9, 12] ✓")
print()

print(f"u * v (element-wise) = {u * v}")
print(f"  Controle: [1×5, 2×6, 3×7, 4×8] = [5, 12, 21, 32] ✓")
print(f"  Let op: dit is NIET het dot product!")