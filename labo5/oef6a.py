import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Implementeer de volgende activatiefuncties en hun afgeleiden:
# 1. ReLU: f(x) = max(0, x)
# 2. Sigmoid: σ(x) = 1/(1 + e^(-x)), afgeleide: σ’(x) = σ(x)(1 - σ(x))
# 3. Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)), afgeleide: tanh’(x) = 1 - tanh²(x)

# Plot elke functie samen met zijn afgeleide.
# Definieer het bereik van x
x = np.linspace(-10, 10, 400)

# Bereken de waarde van de activatiefuncties
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Bereken de waarde van de afgeleiden
relu_prime = np.where(x > 0, 1, 0)
sigmoid_prime = sigmoid * (1 - sigmoid)
tanh_prime = 1 - tanh**2

# Plot de activatiefuncties en hun afgeleiden
plt.plot(x, relu, label="ReLU")
plt.plot(x, relu_prime, label="ReLU'")
plt.plot(x, sigmoid, label="Sigmoid")
plt.plot(x, sigmoid_prime, label="Sigmoid'")
plt.plot(x, tanh, label="Tanh")
plt.plot(x, tanh_prime, label="Tanh'")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()