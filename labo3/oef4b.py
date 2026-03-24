# Voer deze cel eerst uit om alle benodigde libraries te importeren

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

np.set_printoptions(precision=3, suppress=True)

# Laad MNIST
print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

def create_batches(X, batch_size):
    """
    Verdeelt dataset X in batches van grootte batch_size.
    Laatste batch kan kleiner zijn.
    
    Args:
        X: numpy array van vorm (n_samples, features)
        batch_size: grootte van elke batch
    
    Returns:
        lijst van numpy arrays, elke batch is (batch_size, features)
    """
    batches = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        batches.append(batch)
    return batches

# Test
X_subset = X[:100]  # 100 afbeeldingen
batch_size = 32

batches = create_batches(X_subset, batch_size)

print(f"Aantal batches: {len(batches)}")
print(f"Vorm eerste batch: {batches[0].shape}")      # (32, 784)
print(f"Vorm laatste batch: {batches[-1].shape}")    # (4, 784)
print(f"Totaal afbeeldingen: {sum(len(b) for b in batches)}")  # 100 ✓

# Details laatste batch
print(f"\nLaatste batch grootte: {len(batches[-1])}")
print("Batch groottes:", [len(b) for b in batches])

