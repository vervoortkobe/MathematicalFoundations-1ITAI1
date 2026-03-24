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

# Opdracht 7b
similarity_matrix = np.zeros((10, 10))

for i in range(10):
    for j in range(10):
        similarity_matrix[i, j] = cosine_similarity(gemiddelde_cijfers[i], gemiddelde_cijfers[j])

plt.figure(figsize=(8, 7))
im = plt.imshow(similarity_matrix, cmap='YlOrRd', vmin=0.5, vmax=1)
plt.colorbar(im, label='Cosine Similarity')
plt.xticks(range(10))
plt.yticks(range(10))
plt.xlabel('Cijfer')
plt.ylabel('Cijfer')
plt.title('Cosine Similarity tussen Gemiddelde Cijfers')

# Voeg waarden toe
for i in range(10):
    for j in range(10):
        color = 'white' if similarity_matrix[i, j] > 0.75 else 'black'
        plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                ha='center', va='center', color=color, fontsize=8)

plt.tight_layout()
plt.show()