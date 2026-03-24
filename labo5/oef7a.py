import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Beschouw een neuron met meerdere inputs: y = σ(w₁x₁ + w₂x₂ + b).
# Bereken analytisch:
# ∂y/∂w₁
# ∂y/∂w₂
# ∂y/∂b
# ∂y/∂x₁
# ∂y/∂x₂

# Implementeer een functie die al deze gradiënten berekent en verifieer numeriek.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradients(x1, x2, w1, w2, b):
    # Bereken de uitvoer van de neuron
    z = w1 * x1 + w2 * x2 + b
    y = sigmoid(z)

    # Bereken de gradiënten analytisch
    dy_dw1 = x1 * y * (1 - y)
    dy_dw2 = x2 * y * (1 - y)
    dy_db = y * (1 - y)
    dy_dx1 = w1 * y * (1 - y)
    dy_dx2 = w2 * y * (1 - y)

    return dy_dw1, dy_dw2, dy_db, dy_dx1, dy_dx2

# Test de functie met voorbeeldwaarden
x1 = 0.5
x2 = 0.3
w1 = 0.8
w2 = 0.6
b = 0.1
gradients = compute_gradients(x1, x2, w1, w2, b)
print("Analytische gradiënten:")
print(f"∂y/∂w₁: {gradients[0]:.4f}")
print(f"∂y/∂w₂: {gradients[1]:.4f}")
print(f"∂y/∂b: {gradients[2]:.4f}")
print(f"∂y/∂x₁: {gradients[3]:.4f}")
print(f"∂y/∂x₂: {gradients[4]:.4f}")

# Verifieer numeriek
epsilon = 1e-5
def numerical_gradient(func, var, epsilon=1e-5):
    original_value = var
    var += epsilon
    plus_epsilon = func()
    var -= 2 * epsilon
    minus_epsilon = func()
    var = original_value  # Reset variable
    return (plus_epsilon - minus_epsilon) / (2 * epsilon)

def numerical_gradients(x1, x2, w1, w2, b):
    def func_w1():
        return sigmoid(w1 * x1 + w2 * x2 + b)
    def func_w2():
        return sigmoid(w1 * x1 + w2 * x2 + b)
    def func_b():
        return sigmoid(w1 * x1 + w2 * x2 + b)
    def func_x1():
        return sigmoid(w1 * x1 + w2 * x2 + b)
    def func_x2():
        return sigmoid(w1 * x1 + w2 * x2 + b)

    dy_dw1 = numerical_gradient(func_w1, w1, epsilon)
    dy_dw2 = numerical_gradient(func_w2, w2, epsilon)
    dy_db = numerical_gradient(func_b, b, epsilon)
    dy_dx1 = numerical_gradient(func_x1, x1, epsilon)
    dy_dx2 = numerical_gradient(func_x2, x2, epsilon)

    return dy_dw1, dy_dw2, dy_db, dy_dx1, dy_dx2

numerical_grads = numerical_gradients(x1, x2, w1, w2, b)
print("\nNumerieke gradiënten:")
print(f"∂y/∂w₁: {numerical_grads[0]:.4f}")
print(f"∂y/∂w₂: {numerical_grads[1]:.4f}")
print(f"∂y/∂b: {numerical_grads[2]:.4f}")
print(f"∂y/∂x₁: {numerical_grads[3]:.4f}")
print(f"∂y/∂x₂: {numerical_grads[4]:.4f}")