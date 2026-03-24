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

# Opdracht 4b
print("Opdracht 4b: Normalisatie\n")

for i, v in enumerate([v1, v2, v3], 1):
    norm = np.linalg.norm(v)
    v_normalized = v / norm
    norm_check = np.linalg.norm(v_normalized)
    
    print(f"v{i} = {v}")
    print(f"  Genormaliseerd: {v_normalized}")
    print(f"  Norm van genormaliseerd: {norm_check:.10f}")
    print()