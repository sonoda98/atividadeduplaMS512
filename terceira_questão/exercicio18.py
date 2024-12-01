import numpy as np

# Matriz A
A = np.array([[1, 1, 1],
              [-1, 9, 2],
              [0, -1, 2]])

# Shift p
p = 9

# Função para Iteração Inversa
def inverse_iteration(A, q0, p, num_iterations):
    I = np.eye(A.shape[0])  # Matriz identidade
    B = np.linalg.inv(A - p * I)  # (A - pI)^(-1)
    q = q0 / np.linalg.norm(q0)  # Normalizando o vetor inicial
    qs = [q]  # Para armazenar os vetores iterados
    for _ in range(num_iterations):
        z = B @ q  # Multiplicação pela matriz B
        q = z / np.linalg.norm(z)  # Normalização
        qs.append(q)
    return qs

# Parâmetros da Iteração Inversa
q0 = np.array([1, 1, 1])
num_iterations = 10

# Item a
print("Item a)---------------------------------------------------------------")
qs_inverse = inverse_iteration(A, q0, p, num_iterations)

# Imprimindo os iterados
for i, q in enumerate(qs_inverse):
    print(f"Iteração {i}: {q}")
print()

# Item b
print("Item b)---------------------------------------------------------------")
# Calculando autovalores e autovetores
eigvals, eigvecs = np.linalg.eig(A)

# Identificando o autovalor e autovetor dominante
dominant_index = np.argmin(np.abs(eigvals - p))  # Perto de 9
dominant_eigvec = eigvecs[:, dominant_index]
dominant_eigvec /= np.linalg.norm(dominant_eigvec)  # Normalizando

print("Autovalores:")
print(eigvals)
print()
print("Autovetor dominante (normalizado):")
print(dominant_eigvec)
print()

# Calculando erros e razões
errors = [np.linalg.norm(q - dominant_eigvec) for q in qs_inverse]
ratios = [
    errors[j + 1] / errors[j] if errors[j] != 0 else np.inf
    for j in range(len(errors) - 1)
]

print("Erros ||q_j - v||:")
print(errors)
print()
print("Razões ||q_{j+1} - v|| / ||q_j - v||:")
print(ratios)
print()

# Taxa de convergência teórica
sorted_eigvals = np.sort(np.abs(eigvals))
lambda_2 = sorted_eigvals[1]  # Segundo autovalor mais próximo de p
lambda_1 = sorted_eigvals[0]  # Autovalor mais próximo de p
convergence_rate = abs(lambda_2 / (lambda_1 - p))

print(f"Taxa de convergência teórica |lambda_2 / (lambda_1 - p)|: {convergence_rate}")
print()

# Comparação com as razões calculadas
print("Comparação da taxa de convergência teórica com as razões calculadas:")
print(f"Taxa de convergência teórica: {convergence_rate}")
print(f"Última razão calculada: {ratios[-1]}")
