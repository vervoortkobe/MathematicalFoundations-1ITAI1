import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Implementeer de softmax functie en bewijs dat de output altijd:
# 1. Positief is
# 2. Sommeert tot 1
# Test met verschillende inputs, inclusief grote en kleine waarden.

def softmax(x):
    """Numeriek stabiele softmax-functie."""
    # Controleer of er oneindige waarden in de input zitten
    if np.any(np.isinf(x)):
        # Vervang oneindigheden door een zeer grote, maar eindige waarde
        x = np.where(np.isinf(x), 700, x)  # 700 is voldoende groot om exp(x) te laten overstromen, maar niet oneindig

    # Trek de maximale waarde af voor numerieke stabiliteit
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Test
test_inputs = [
    np.array([1, 2, 3]),
    np.array([100, 200, 300]),
    np.array([-1, -2, -3]),
    np.array([0, 0, 0]),
    np.array([10, 10, 10]),
    np.array([1e10, 1e100, 1e100])
]

for i, x in enumerate(test_inputs):
    output = softmax(x)
    print(f"Input {i+1}: {x}")
    print(f"Softmax output: {output}")
    print(f"Alle waarden positief: {all(output > 0)}")
    print(f"Sommeert tot 1: {np.isclose(np.sum(output), 1)}")
    print()