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

# Breid uit naar een neuron met ReLU activatie: y = ReLU(w₁x₁ + w₂x₂ + b).
# Bereken de gradiënten en merk op dat ze afhankelijk zijn van het teken van de input naar ReLU.
def relu(z):
    return np.maximum(0, z)

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

def compute_gradients_relu(x1, x2, w1, w2, b):
  # Bereken de uitvoer van de neuron
  z = w1 * x1 + w2 * x2 + b
  y = relu(z)

  # Bereken de gradiënten analytisch
  # ReLU derivative is 1 if z > 0, else 0
  dy_dz = 1.0 if z > 0 else 0.0
  
  dy_dw1 = x1 * dy_dz
  dy_dw2 = x2 * dy_dz
  dy_db = dy_dz
  dy_dx1 = w1 * dy_dz
  dy_dx2 = w2 * dy_dz

  return dy_dw1, dy_dw2, dy_db, dy_dx1, dy_dx2

# Test de functie met voorbeeldwaarden
x1 = 0.5
x2 = 0.3
w1 = 0.8
w2 = 0.6
b = 0.1
gradients = compute_gradients(x1, x2, w1, w2, b)
print("Analytische gradiënten (Sigmoid):")
print(f"∂y/∂w₁: {gradients[0]:.4f}")
print(f"∂y/∂w₂: {gradients[1]:.4f}")
print(f"∂y/∂b: {gradients[2]:.4f}")
print(f"∂y/∂x₁: {gradients[3]:.4f}")
print(f"∂y/∂x₂: {gradients[4]:.4f}")

gradients_relu = compute_gradients_relu(x1, x2, w1, w2, b)
print("\nAnalytische gradiënten (ReLU):")
print(f"∂y/∂w₁: {gradients_relu[0]:.4f}")
print(f"∂y/∂w₂: {gradients_relu[1]:.4f}")
print(f"∂y/∂b: {gradients_relu[2]:.4f}")
print(f"∂y/∂x₁: {gradients_relu[3]:.4f}")
print(f"∂y/∂x₂: {gradients_relu[4]:.4f}")

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

def numerical_gradients(x1, x2, w1, w2, b, activation='sigmoid'):
  act_func = sigmoid if activation == 'sigmoid' else relu
  
  def func_w1():
    return act_func(w1 * x1 + w2 * x2 + b)
  def func_w2():
    return act_func(w1 * x1 + w2 * x2 + b)
  def func_b():
    return act_func(w1 * x1 + w2 * x2 + b)
  def func_x1():
    return act_func(w1 * x1 + w2 * x2 + b)
  def func_x2():
    return act_func(w1 * x1 + w2 * x2 + b)

  dy_dw1 = numerical_gradient(func_w1, w1, epsilon)
  dy_dw2 = numerical_gradient(func_w2, w2, epsilon)
  dy_db = numerical_gradient(func_b, b, epsilon)
  dy_dx1 = numerical_gradient(func_x1, x1, epsilon)
  dy_dx2 = numerical_gradient(func_x2, x2, epsilon)

  return dy_dw1, dy_dw2, dy_db, dy_dx1, dy_dx2

numerical_grads = numerical_gradients(x1, x2, w1, w2, b, 'sigmoid')
print("\nNumerieke gradiënten (Sigmoid):")
print(f"∂y/∂w₁: {numerical_grads[0]:.4f}")
print(f"∂y/∂w₂: {numerical_grads[1]:.4f}")
print(f"∂y/∂b: {numerical_grads[2]:.4f}")
print(f"∂y/∂x₁: {numerical_grads[3]:.4f}")
print(f"∂y/∂x₂: {numerical_grads[4]:.4f}")

numerical_grads_relu = numerical_gradients(x1, x2, w1, w2, b, 'relu')
print("\nNumerieke gradiënten (ReLU):")
print(f"∂y/∂w₁: {numerical_grads_relu[0]:.4f}")
print(f"∂y/∂w₂: {numerical_grads_relu[1]:.4f}")
print(f"∂y/∂b: {numerical_grads_relu[2]:.4f}")
print(f"∂y/∂x₁: {numerical_grads_relu[3]:.4f}")
print(f"∂y/∂x₂: {numerical_grads_relu[4]:.4f}")