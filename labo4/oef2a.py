import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Stelsel 1:
# 2x + y = 5
# x - y = 1
A = [[2, 1], [1, -1]]
b = [5, 1]

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

print()

# Stelsel 2:
# x + y + z = 6
# 2x - y + z = 3
# x + 2y - z = 3
A = [[1, 1, 1], [2, -1, 1], [1, 2, -1]]
b = [6, 3, 3]

# Oplossen met NumPy
x_oplossing = np.linalg.solve(A, b)

print(f"Oplossing: x = {x_oplossing}")
print(f"  x = {x_oplossing[0]}")
print(f"  y = {x_oplossing[1]}")
print(f"  z = {x_oplossing[2]}")
print()

# Verificatie
print("Verificatie (A @ x moet gelijk zijn aan b):")
print(f"  A @ x = {A @ x_oplossing}")
print(f"  b     = {b}")
print(f"  Gelijk? {np.allclose(A @ x_oplossing, b)}")

