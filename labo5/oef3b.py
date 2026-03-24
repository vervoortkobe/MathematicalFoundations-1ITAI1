import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Beschouw de functie f(x) = σ(2x - 1) waarbij σ de sigmoid functie is.
# 1. Bereken f’(x) met de kettingregel
# 2. Implementeer f(x) en f’(x)
# 3. Plot beide functies
# 4. Verifieer numeriek op x = 0, 0.5, 1

# f'(x) = σ'(2x - 1) * 2
# σ'(z) = σ(z) * (1 - σ(z))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def numerical_derivative(f, x, h=1e-5):
  return (f(x + h) - f(x)) / h

def f(x):
  return sigmoid(2*x - 1)

def f_prime(x):
  return sigmoid(2*x - 1) * 2 * (1 - sigmoid(2*x - 1))

x_values = np.linspace(-1, 2, 100)
plt.plot(x_values, f(x_values), label='f(x)')
plt.plot(x_values, f_prime(x_values), label="f'(x)")
plt.legend()
plt.title("f(x) en f'(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

test_points = [0, 0.5, 1]
for x in test_points:
  print(f"f'({x}) = {f_prime(x):.4f}, numeriek: {numerical_derivative(f, x):.4f}")