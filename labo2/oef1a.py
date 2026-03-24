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

# Opdracht 1a
print("Opdracht 1a: Vectoren aanmaken\n")

# 1. Getallen 0-9
v1 = np.arange(10)
print(f"1. np.arange(10): {v1}")

# 2. Vijf nullen
v2 = np.zeros(5)
print(f"2. np.zeros(5): {v2}")

# 3. Vier enen
v3 = np.ones(4)
print(f"3. np.ones(4): {v3}")

# 4. Zes getallen gelijk verdeeld tussen 0 en 1
v4 = np.linspace(0, 1, 6)
print(f"4. np.linspace(0, 1, 6): {v4}")