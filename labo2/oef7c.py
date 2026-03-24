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

# Opdracht 7c
print("Opdracht 7c: Analyse van similarity matrix\n")

# Maak lijst van alle paren (zonder diagonaal)
paren = []
for i in range(10):
    for j in range(i+1, 10):
        paren.append((i, j, similarity_matrix[i, j]))

paren_gesorteerd = sorted(paren, key=lambda x: x[2], reverse=True)

# 1. Meest gelijkend
meest_gelijkend = paren_gesorteerd[0]
print(f"1. Meest gelijkende paar: {meest_gelijkend[0]} en {meest_gelijkend[1]}")
print(f"   Similarity: {meest_gelijkend[2]:.4f}")

# 2. Minst gelijkend
minst_gelijkend = paren_gesorteerd[-1]
print(f"\n2. Minst gelijkende paar: {minst_gelijkend[0]} en {minst_gelijkend[1]}")
print(f"   Similarity: {minst_gelijkend[2]:.4f}")

# 3. Cijfer met laagste gemiddelde similarity
gem_sim_per_cijfer = []
for i in range(10):
    # Gemiddelde similarity met alle andere cijfers
    sim_anderen = [similarity_matrix[i, j] for j in range(10) if i != j]
    gem_sim_per_cijfer.append(np.mean(sim_anderen))

meest_uniek = np.argmin(gem_sim_per_cijfer)
print(f"\n3. Meest unieke cijfer: {meest_uniek}")
print(f"   Gemiddelde similarity met andere cijfers: {gem_sim_per_cijfer[meest_uniek]:.4f}")