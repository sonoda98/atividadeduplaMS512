import numpy as np

# Parte (a): Método das potências com a nova matriz A
A = np.array([[1, 1, 1], [-1, 3, 2], [-4, -1, 2]], dtype=float)
q = np.ones((3, 1))  # Vetor inicial q_0
iteracoes = 10  # Número de iterações
iteracao = np.zeros((3, iteracoes + 1))
scale_factors = []

for j in range(iteracoes):
    iteracao[:, j] = q.flatten()  # Armazena q_j (vetor coluna como linha para compatibilidade)
    q = np.dot(A, q)  # Multiplicação por A
    scale_factor = np.max(np.abs(q))  # Maior valor absoluto
    scale_factors.append(scale_factor)
    q = q / scale_factor  # Normaliza o vetor

iteracao[:, -1] = q.flatten()  # Último vetor após 10 iterações

# Parte (b): Cálculo dos autovalores e autovetores de A
autovalores, autovetores = np.linalg.eig(A)
autovalor_dominante = autovalores[np.argmax(np.abs(autovalores))]  # Autovalor dominante
autovetor_dominante = autovetores[:, np.argmax(np.abs(autovalores))]  # Autovetor dominante

# Normalizando o autovetor dominante para comparação
autovetor_dominante /= np.linalg.norm(autovetor_dominante)

# Parte (c): Razões de erros para cada iteração
taxa_erro = []
erro_inicial = np.linalg.norm(iteracao[:, 0] - autovetor_dominante)

for j in range(1, iteracoes):
    q_next = iteracao[:, j]
    erro = np.linalg.norm(q_next - autovetor_dominante)
    taxa_erro.append((erro / erro_inicial) ** (1 / j))

# Parte (d): Calcula o quociente dos autovalores dominantes
autovalores_ordenados = np.sort(np.abs(autovalores))  # Ordena os autovalores em valor absoluto
lambda_ratio = autovalores_ordenados[-2] / autovalores_ordenados[-1]  # |λ2/λ1|

# Resultados
print("Iterações (vetores normalizados):")
print(iteracao)
print("\nAutovalor dominante:")
print(autovalor_dominante)
print("\nAutovetor dominante (normalizado):")
print(autovetor_dominante)
print("\nTaxas de erro em cada iteração:")
print(taxa_erro)
print("\nRazão entre os dois maiores autovalores:")
print(lambda_ratio)
