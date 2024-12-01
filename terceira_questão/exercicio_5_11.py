import numpy as np

# Parte (a): Método das potências com a nova matriz A
A = np.array([[1, 1, 1], [-1, 9, 2], [-4, -1, 2]], dtype=float)
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

# Parte (b): Calcula autovalores e autovetores de A
autovalores, autovetores = np.linalg.eig(A)
autovalor_dominante = np.max(autovalores)  # Autovalor dominante
autovetor_dominante = autovetores[:, np.argmax(autovalores)]  # Autovetor dominante

# Normalizando o autovetor dominante para comparação
autovetor_dominante /= np.linalg.norm(autovetor_dominante)

# Parte (c): Calculo da métrica para cada iteração j
taxa_erro = []
erro_inicial = np.linalg.norm(iteracao[:, 1] - autovetor_dominante)

for j in range(1, iteracoes):
    q_next = iteracao[:, j + 1]
    erro = np.linalg.norm(q_next - autovetor_dominante)
    taxa_erro.append(erro ** (1 / j) / erro_inicial ** (1 / j))

# Parte (d): Calculo do quociente dos autovalores dominantes
autovalores_ordenados = np.sort(np.abs(autovalores))  # Ordena os autovalores em valor absoluto
quociente = autovalores_ordenados[-2] / autovalores_ordenados[-1]  # |λ2/λ1|

print(iteracao)
print(autovalor_dominante)
print(autovetor_dominante)
print(taxa_erro)
print(quociente)