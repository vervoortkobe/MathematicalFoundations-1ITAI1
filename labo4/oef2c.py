import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# x + 2y = 3
# 2x + 4y = 8
A = [[1, 2], [2, 4]]
b = [3, 8]

x_oplossing = np.linalg.solve(A, b)

print(f"Oplossing: x = {x_oplossing}")
print(f"  x = {x_oplossing[0]}")
print(f"  y = {x_oplossing[1]}")
print()

# Verificatie
print("Verificatie (A @ x moet gelijk zijn aan b):")
print(f"  A @ x = {A @ x_oplossing}")
print(f"  b     = {b}")
print(f"  Gelijk? {np.allclose(A @ x_oplossing, b)}")

# Leg uit waarom dit stelsel geen oplossing heeft.
"""
numpy.linalg.LinAlgError: Singular matrix
Dit stelsel heeft geen unieke oplossing omdat de matrix singulier is, met determinant nul. 
De tweede vergelijking is precies dubbel zo groot als de eerste, maar de rechterkant klopt niet (2*3 = 6 en niet 8), wat een contradictie veroorzaakt.
"""
