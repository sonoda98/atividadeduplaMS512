import numpy as np

# Parte (a): Implementação do método das potências
A = np.array([[1, 1, 1], [-1, 9, 2], [0, -1, 2]], dtype=float)
q = np.ones((3, 1))  # Vetor inicial q_0
iteracoes = 10  # Número de iterações
iteracao = np.zeros((3, iteracoes + 1))
scale_factors = []

for j in range(iteracoes):
    iteracao[:, [j]] = q  # Armazena q_j
    q = np.dot(A, q)  # Multiplicação por A
    scale_factor = np.max(np.abs(q))  # Maior valor absoluto
    scale_factors.append(scale_factor)
    q = q / scale_factor  # Normaliza o vetor

iteracao[:, [-1]] = q  # Último vetor após 10 iterações

# Parte (b): Calculo autovalores e autovetores de A
autovalores, autovetores = np.linalg.eig(A)
autovalor_dominante = np.max(autovalores)  # Autovalor dominante
autovetor_dominante = autovetores[:, np.argmax(autovalores)]  # Autovetor dominante

# Normalizando o autovetor dominante para comparação
autovetor_dominante /= np.linalg.norm(autovetor_dominante)

# Parte (c): Calculo da razão da norma para iterados q_j e autovetor dominante
razao_norma = []
for j in range(1, iteracoes):
    q_j = iteracao[:, j]
    q_next = iteracao[:, j + 1]
    razao_norma.append(np.linalg.norm(q_next - autovetor_dominante) /
                       np.linalg.norm(q_j - autovetor_dominante))

# Parte (d): Calculo do quociente dos autovalores dominantes
autovalores_ordenados = np.sort(np.abs(autovalores))  # Ordena os autovalores em valor absoluto
quociente = autovalores_ordenados[-2] / autovalores_ordenados[-1]  # |λ2/λ1|

print(iteracao)
print(autovalor_dominante)
print(autovetor_dominante)
print(razao_norma)
print(quociente)