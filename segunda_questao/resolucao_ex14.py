import numpy as np

# Criando uma matriz A (n x m) e um vetor b
np.random.seed(42)
n, m = 5, 3  # Dimensão de A (n > m)
A = np.random.rand(n, m)
b = np.random.rand(n)

# Calculando a pseudoinversa de A
A_pseudo_inv = np.linalg.pinv(A)

# Solução de norma mínima
x = A_pseudo_inv @ b

# Verificando a norma mínima (resíduo ||b - Ax||_2)
residuo = np.linalg.norm(b - A @ x, ord=2)

print(A_pseudo_inv)
print(x)
print(residuo)