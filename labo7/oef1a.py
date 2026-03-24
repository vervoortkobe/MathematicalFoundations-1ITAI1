import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Implementeer de volgende operaties met forward en backward methodes:
# 1. Add: z = x + y
# 2. Multiply: z = x * y
# 3. Power: z = x^n

class Add:
    def forward(self, x, y):
        # sla inputs op voor backward
        self.x = x
        self.y = y
        # z = x + y
        return x + y
    
    def backward(self, dout):
        # dz/dx = 1, dz/dy = 1  ⇒  dL/dx = dout * 1, dL/dy = dout * 1 [web:7]
        dx = dout
        dy = dout
        return dx, dy


class Multiply:
    def forward(self, x, y):
        self.x = x
        self.y = y
        # z = x * y
        return x * y
    
    def backward(self, dout):
        # z = x * y
        # dz/dx = y, dz/dy = x  ⇒  dL/dx = dout * y, dL/dy = dout * x [web:7]
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class Power:
    def __init__(self, n):
        self.n = n
    
    def forward(self, x):
        self.x = x
        # z = x^n
        return x ** self.n
    
    def backward(self, dout):
        # z = x^n ⇒ dz/dx = n * x^(n-1)
        # dL/dx = dout * n * x^(n-1)
        dx = dout * self.n * (self.x ** (self.n - 1))
        return dx

# Test

add = Add()
mul = Multiply()
pow2 = Power(2)

x = np.array([2.0])
y = np.array([3.0])

print("=== Add ===")
z_add = add.forward(x, y)
dx_add, dy_add = add.backward(np.array([1.0]))
print("z =", z_add)          # verwacht 5
print("dz/dx =", dx_add)     # verwacht 1
print("dz/dy =", dy_add)     # verwacht 1

print("\n=== Multiply ===")
z_mul = mul.forward(x, y)
dx_mul, dy_mul = mul.backward(np.array([1.0]))
print("z =", z_mul)          # verwacht 6
print("dz/dx =", dx_mul)     # verwacht y = 3
print("dz/dy =", dy_mul)     # verwacht x = 2

print("\n=== Power (n=2) ===")
z_pow = pow2.forward(x)
dx_pow = pow2.backward(np.array([1.0]))
print("z =", z_pow)          # verwacht 4
print("dz/dx =", dx_pow)     # verwacht 2 * x = 4