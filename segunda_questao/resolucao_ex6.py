import numpy as np

# Criando uma matriz aleatória A
np.random.seed(42)
A = np.random.rand(4, 4)

# Calculando os valores singulares de A
valores_singulares = np.linalg.svd(A, compute_uv=False)
sigma_1 = max(valores_singulares)  # Maior valor singular
sigma_n = min(valores_singulares)  # Menor valor singular

# Calculando a norma 2 de A e de A^-1
norma_A = np.linalg.norm(A, ord=2)
norma_A_inv = np.linalg.norm(np.linalg.inv(A), ord=2)

# Calculando o número de condição
num_cond = norma_A * norma_A_inv

# Verificando se num_cond = sigma_1 / sigma_n
num_cond_teorico = sigma_1 / sigma_n

print(num_cond)
print(num_cond_teorico)