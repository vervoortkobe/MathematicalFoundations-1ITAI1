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

# Opdracht 3c
print("Opdracht 3c: Distributiviteit\n")

c = 2.5
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

links = c * (u + v)
rechts = c*u + c*v

print(f"c = {c}, u = {u}, v = {v}")
print()
print(f"c(u + v) = {c} × {u + v} = {links}")
print(f"cu + cv = {c*u} + {c*v} = {rechts}")
print()
print(f"Distributiviteit geldt: {np.allclose(links, rechts)}")