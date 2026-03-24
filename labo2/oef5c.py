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

# Opdracht 5c
print("Opdracht 5c: Hoeken berekenen\n")

paren = [
    (np.array([1, 0]), np.array([1, 1])),
    (np.array([1, 0]), np.array([0, 1])),
    (np.array([1, 1]), np.array([-1, 1])),
]

for i, (u, v) in enumerate(paren, 1):
    dot = u @ v
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cos_theta = dot / (norm_u * norm_v)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    
    print(f"{i}. u = {u}, v = {v}")
    print(f"   cos(θ) = {dot} / ({norm_u:.3f} × {norm_v:.3f}) = {cos_theta:.4f}")
    print(f"   θ = {theta_deg:.1f}°")
    print()