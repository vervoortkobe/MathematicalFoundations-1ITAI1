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

# Opdracht 2a
print("Opdracht 2a: Vector-matrix conversie\n")

afbeelding_vector = X[500]
afbeelding_matrix = afbeelding_vector.reshape(28, 28)

print(f"Shape als vector: {afbeelding_vector.shape}")
print(f"Shape als matrix: {afbeelding_matrix.shape}")

# Verifieer dat ze dezelfde data bevatten
terug_naar_vector = afbeelding_matrix.flatten()
print(f"\nZelfde data? {np.allclose(afbeelding_vector, terug_naar_vector)}")