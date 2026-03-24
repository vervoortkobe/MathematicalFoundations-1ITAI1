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
  
# Opdracht 1b
print("Opdracht 1b: Indexering en slicing\n")

v = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print(f"Vector v: {v}\n")

# 1. Derde element (index 2)
print(f"1. Derde element (v[2]): {v[2]}")

# 2. Laatste twee elementen
print(f"2. Laatste twee (v[-2:]): {v[-2:]}")

# 3. Elk tweede element vanaf het eerste
print(f"3. Elk tweede (v[::2]): {v[::2]}")

# 4. Omgekeerde volgorde
print(f"4. Omgekeerd (v[::-1]): {v[::-1]}")