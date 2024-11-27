import numpy as np

# Definindo a matriz A e o vetor b
A = np.array([[1, 2], [2, 4], [3, 6]])
b = np.array([1, 1, 1])

# Realizando a Decomposição em Valores Singulares (SVD)
U, S, VT = np.linalg.svd(A)

# Construindo a matriz diagonal S+ (pseudoinversa de S)
S_invs = np.zeros((VT.shape[0], U.shape[0]))
for i in range(len(S)):
    if S[i] > 1e-10:  # Considerando valores singulares não nulos
        S_invs[i, i] = 1 / S[i]

# Calculando a pseudoinversa de A
A_invs = VT.T @ S_invs @ U.T

# Solução de mínimos quadrados de norma mínima
x = A_invs @ b

print("Solução de quadrados mínimos da norma mínima:")
print(x)