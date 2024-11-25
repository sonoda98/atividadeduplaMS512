import numpy as np
from numpy import array
from scipy.linalg import svd

# Matriz A e vetor b definidos no problema
A = np.array([[1, 2], [2, 4], [3, 6]])
b = np.array([1, 1, 1])

# Decomposição SVD de A
U, Sigma, VT = np.linalg.svd(A, full_matrices=False)

# Construção da pseudoinversa de A a partir de SVD
Sigma_pseudo = np.zeros_like(A.T)
np.fill_diagonal(Sigma_pseudo, 1 / Sigma)  # Invertendo os valores singulares (não zero)
A_pseudo_svd = VT.T @ Sigma_pseudo @ U.T

# Solução de norma mínima
x_svd = A_pseudo_svd @ b

# Verificando a norma do resíduo ||b - Ax||_2
residuo_svd = np.linalg.norm(b - A @ x_svd, ord=2)

x_svd, residuo_svd
