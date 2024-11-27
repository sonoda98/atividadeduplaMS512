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

print("Item d)----------------------------------------------------------------")
# Base alternativa
phi2_factor = 3 * 10**7
shift = 0.98765435

# Criando a matriz Â
rows_alt = []
for i in range(len(data)):
    t = data[i, 0]
    new_row_alt = [1, phi2_factor * (t - shift)]  # Base [1, 3*10^7*(t - 0.98765435)]
    rows_alt.append(new_row_alt)
A_alt = np.array(rows_alt)

# Número de condicionamento de Â
cond_A_alt = np.linalg.cond(A_alt)
print("Número de condicionamento de Â:")
print(cond_A_alt)
print()

# Fatoração QR para Â
Q_alt, R_alt = np.linalg.qr(A_alt)
Qt_b_alt = np.dot(Q_alt.T, b)
xQR_alt = np.linalg.solve(R_alt, Qt_b_alt)
print("Solução QR com base alternativa (x):")
print(xQR_alt)
print()

# Norma do resíduo para QR
residuoQR_alt = np.linalg.norm(A_alt @ xQR_alt - b)
print("Norma do resíduo (QR - base alternativa):")
print(residuoQR_alt)
print()

print("Item e)----------------------------------------------------------------")
# Obtendo as equações normais para Â
AtA_alt = np.dot(A_alt.T, A_alt)
Atb_alt = np.dot(A_alt.T, b)
cond_AtA_alt = np.linalg.cond(AtA_alt)
print("Número de condicionamento de Â^T Â:")
print(cond_AtA_alt)
print()

xNormal_alt = np.linalg.solve(AtA_alt, Atb_alt)
print("Solução eq normal (base alternativa - x):")
print(xNormal_alt)
print()

# Norma do resíduo para as equações normais
residuoNormal_alt = np.linalg.norm(A_alt @ xNormal_alt - b)
print("Norma do resíduo (eq normal - base alternativa):")
print(residuoNormal_alt)
print()
