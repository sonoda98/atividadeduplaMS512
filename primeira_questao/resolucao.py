import numpy as np

#Leitura do csv
data = np.loadtxt("./data.csv", delimiter=',')

print("Item a)----------------------------------------------------------------")
# Criando a matriz A
rows = []
for i in range(len(data)):
	new_row = np.array([1, data[i][0]])#Assumindo que a base é [1, t]
	rows.append(new_row)
A = np.array(rows)
print("Matriz A: (base [1, t]):")
print(A)
print()

#Criando o vetor b
b = data[:, [1]]
print("Vetor b:")
print(b)
print()

#Número de condicionamento de A
cond_number = np.linalg.cond(A)
print("Número de condicionamento de A:")
print(cond_number)
print()

#Fatoração QR
Q, R = np.linalg.qr(A)
print("Matriz Q (ortogonal):")
print(Q)
print()
print("Matriz R (triangular superior):")
print(R)
print()

#Resolver R x = Q^T b
Qt_b = np.dot(Q.T, b)
x = np.linalg.solve(R, Qt_b)
print("Solução de mínimos quadrados (x):")
print(x)
print()

#Norma do resíduo
residuo = np.linalg.norm(A @ x - b)
print("Norma do residuo:")
print(residuo)
print()

print("Item b)----------------------------------------------------------------")
#Calculando A^T A e A^T b