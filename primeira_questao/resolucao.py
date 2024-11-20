import numpy as np
import matplotlib.pyplot as plt

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
cond_A = np.linalg.cond(A)
print("Número de condicionamento de A:")
print(cond_A)
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
xQR = np.linalg.solve(R, Qt_b)
print("Solução de mínimos quadrados (x):")
print(xQR)
print()

#Norma do resíduo
residuoQR = np.linalg.norm(A @ xQR - b)
print("Norma do residuo:")
print(residuoQR)
print()

print("Item b)----------------------------------------------------------------")
#Calculando A^T A e A^T b
AtA = np.dot(A.T, A)
Atb = np.dot(A.T, b)
print("Matriz A^T A (coeficientes):")
print(AtA)
print()
print("Vetor A^T b (lado direito):")
print(Atb)
print()

#Número de cond de AtA
cond_AtA = np.linalg.cond(AtA)
print("Número de condicionamento da matriz A^T A:")
print(cond_AtA)
print()

#Resolver eq normal
xNormal = np.linalg.solve(AtA, Atb)
print("Solução eq normal (x):")
print(xNormal)
print()

#Norma do resíduo
residuoNormal = np.linalg.norm(A @ xNormal - b)
print("Norma do residuo (eq normal):")
print(residuoNormal)
print()

print("Item c)----------------------------------------------------------------")
#Extraindo colunas dos dados
t = data[:, 0]
y = data[:, 1]

#Gerando pontos de x para as retas
x_vals = np.linspace(0.9876543, 0.9876544, 10)

#Gerando os pontos de y para cada um dos métodos
yQR = xQR[0] + xQR[1] * x_vals
yNormal = xNormal[0] + xNormal[1] * x_vals


plt.plot(t, y, 'o', label="Data", color="black")

plt.plot(x_vals, yQR, label="Solução QR", linestyle='-', color="blue")

plt.plot(x_vals, yNormal, label="Solução Normal", linestyle='--', color="red")

plt.legend(loc="best")
plt.show()
