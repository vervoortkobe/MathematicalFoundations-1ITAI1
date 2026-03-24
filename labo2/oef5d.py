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

# Opdracht 5d
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (u, v) in zip(axes, paren):
    # Bereken hoek
    cos_theta = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
    theta_deg = np.degrees(np.arccos(cos_theta))
    
    # Teken vectoren
    ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.03, label='u')
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
              color='red', width=0.03, label='v')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title(f'θ = {theta_deg:.1f}°')

plt.tight_layout()
plt.show()