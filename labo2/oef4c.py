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

# Opdracht 4c
np.random.seed(456)
indices = np.random.choice(len(X), 5, replace=False)

normen = [(idx, np.linalg.norm(X[idx])) for idx in indices]
normen_gesorteerd = sorted(normen, key=lambda x: x[1])

print("Normen van 5 willekeurige MNIST afbeeldingen:\n")
for idx, norm in normen:
    print(f"  Index {idx} (label {y[idx]}): norm = {norm:.2f}")

kleinste_idx = normen_gesorteerd[0][0]
grootste_idx = normen_gesorteerd[-1][0]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(X[kleinste_idx].reshape(28, 28), cmap='gray')
axes[0].set_title(f'Kleinste norm: {normen_gesorteerd[0][1]:.2f}\nLabel: {y[kleinste_idx]}')
axes[0].axis('off')

axes[1].imshow(X[grootste_idx].reshape(28, 28), cmap='gray')
axes[1].set_title(f'Grootste norm: {normen_gesorteerd[-1][1]:.2f}\nLabel: {y[grootste_idx]}')
axes[1].axis('off')

plt.tight_layout()
plt.show()

print("\nDe norm is groter als het cijfer meer witte pixels heeft.")
print("Een dik geschreven cijfer heeft een hogere norm dan een dun cijfer.")