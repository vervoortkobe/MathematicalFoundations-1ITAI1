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

# 1. (3×2) @ (2×4)
# (3x4)
# 2. (2×3) @ (2×3)
# product niet gedifinieerd
# 3. (4×1) @ (1×3)
# (4x3)
# 4. (5×3) @ (3×5)
# (5x3)
# 5. (2×2) @ (2×2) @ (2×2)
# (2x2)

# Regel: een product A@BA@B is gedefinieerd als het aantal kolommen van AA gelijk is aan het aantal rijen van BB. De shape van het resultaat is dan (rijen van A,kolommen van B)(rijen van A,kolommen van B).
