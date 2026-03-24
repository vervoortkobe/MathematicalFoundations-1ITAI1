# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 1a: Vector met getallen 1 tot 10

vector = np.arange(1, 11)
print(f"Vector: {vector}")
print(f"Aantal elementen: {len(vector)}")
