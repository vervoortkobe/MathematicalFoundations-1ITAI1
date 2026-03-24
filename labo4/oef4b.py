import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Bereken de determinant van de volgende matrices met de hand en verifieer met NumPy:
# det(A) = ad - bc
A = [[3, 8],
     [4, 6]]
# det(A) = 3*6 - 8*4 = 18 - 32 = -14
B = [[1, 2],
     [3, 4]]
# det(B) = 1*4 - 2*3 = 4 - 6 = -2
C = [[2, 4],
     [1, 2]]
# det(C) = 2*2 - 4*1 = 4 - 4 = 0

det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
det_C = np.linalg.det(C)

print(f"Determinant A: {det_A}")
print(f"Determinant B: {det_B}")
print(f"Determinant C: {det_C}")

# Bepaal welke van de matrices uit 4a inverteerbaar zijn op basis van hun determinant. Verifieer door te proberen de inverse te berekenen.
# Een matrix is inverteerbaar als en slechts als de determinant niet nul is.

# det(A) = -14, dus A is inverteerbaar.
print(f"Inverse van A:\n{np.linalg.inv(A)}")
# det(B) = -2, dus B is inverteerbaar.
print(f"Inverse van B:\n{np.linalg.inv(B)}")
# det(C) = 0, dus C is niet inverteerbaar.
try:
    print(f"Inverse van C:\n{np.linalg.inv(C)}")
except np.linalg.LinAlgError:
    print("C is niet inverteerbaar, geen inverse bestaat.")
