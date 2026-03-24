import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Verifieer de volgende eigenschappen van determinanten:
    # det(A × B) = det(A) × det(B)
    # det(Aᵀ) = det(A)
    # det(c × A) = cⁿ × det(A) voor een n×n matrix
# Gebruik willekeurige 3×3 matrices en c = 2.

A = np.random.rand(3, 3)
B = np.random.rand(3, 3)
print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
c = 2
print()

print(f"Klopt det(A × B) = det(A) × det(B)?")
print(f"det(A × B): {np.linalg.det(np.dot(A, B))}")
print(f"det(A) × det(B): {np.linalg.det(A) * np.linalg.det(B)}")
print(f"Zijn ze gelijk? {np.isclose(np.linalg.det(np.dot(A, B)), np.linalg.det(A) * np.linalg.det(B))}")
print()

print(f"Klopt det(Aᵀ) = det(A)?")
print(f"det(Aᵀ): {np.linalg.det(A.T)}")
print(f"det(A): {np.linalg.det(A)}")
print(f"Zijn ze gelijk? {np.isclose(np.linalg.det(A.T), np.linalg.det(A))}")
print()

print(f"Klopt det(c × A) = cⁿ × det(A)?")
n = A.shape[0]  # n is het aantal rijen (of kolommen) van de vierkante matrix
print(f"det(c × A): {np.linalg.det(c * A)}")
print(f"cⁿ × det(A): {c**n * np.linalg.det(A)}")
print(f"Zijn ze gelijk? {np.isclose(np.linalg.det(c * A), c**n * np.linalg.det(A))}")
print()
