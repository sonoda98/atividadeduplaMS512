import numpy as np

# Criando uma matriz aleatória A
np.random.seed(42)
A = np.random.rand(4, 4)

# Calculando os valores singulares de A
singular_values = np.linalg.svd(A, compute_uv=False)
sigma_1 = max(singular_values)  # Maior valor singular
sigma_n = min(singular_values)  # Menor valor singular

# Calculando a norma 2 de A e de A^-1
norm_A = np.linalg.norm(A, ord=2)
norm_A_inv = np.linalg.norm(np.linalg.inv(A), ord=2)

# Calculando o número de condição
kappa_2 = norm_A * norm_A_inv

# Verificando se kappa_2 = sigma_1 / sigma_n
kappa_2_teorico = sigma_1 / sigma_n

print(kappa_2)
print(kappa_2_teorico)