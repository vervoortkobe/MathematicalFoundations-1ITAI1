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

# Bonusopdracht
print("Bonusopdracht: Kleuren MNIST\n")

def colorize_mnist(image_vector, color):
    """Maak een kleurenversie van een MNIST afbeelding."""
    # Normaliseer de afbeelding naar [0, 1]
    img = image_vector.reshape(28, 28) / 255.0
    
    # Maak RGB afbeelding
    rgb = np.zeros((28, 28, 3))
    for i, c in enumerate(color):
        rgb[:, :, i] = img * c
    
    return rgb

# Kleuren: rood, groen, blauw, geel
colors = [
    (1, 0, 0),    # Rood
    (0, 1, 0),    # Groen
    (0, 0, 1),    # Blauw
    (1, 1, 0),    # Geel
]

# Neem een MNIST afbeelding
sample_img = X[0]
print(f"Originele MNIST shape: {sample_img.shape}")

# Maak batch van gekleurde versies
batch = np.array([colorize_mnist(sample_img, c) for c in colors])
print(f"Batch shape: {batch.shape}")
print(f"Dit is een 4D tensor: (batch_size, height, width, channels)")

# Visualiseer
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

# Origineel (grijswaarden)
axes[0].imshow(sample_img.reshape(28, 28), cmap='gray')
axes[0].set_title(f'Origineel\nShape: (28, 28)')
axes[0].axis('off')

# Gekleurde versies
color_names = ['Rood', 'Groen', 'Blauw', 'Geel']
for i, (ax, name) in enumerate(zip(axes[1:], color_names)):
    ax.imshow(batch[i])
    ax.set_title(f'{name}\nShape: (28, 28, 3)')
    ax.axis('off')

plt.suptitle(f'Van grijswaarden naar kleur: dimensies nemen toe!', fontsize=14)
plt.tight_layout()
plt.show()