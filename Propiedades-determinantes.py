import numpy as np

print("=== PROPIEDADES DE LOS DETERMINANTES (2 EJEMPLOS POR PROPIEDAD) ===")

# ============================
# 1. IDENTIDAD
# ============================
print("\n=== PROPIEDAD 1: det(I) = 1 ===")
I2 = np.eye(2)
I3 = np.eye(3)

print("\nEjemplo 1: I 2x2")
print("Matriz:\n", I2)
print("Resultado:", np.linalg.det(I2))

print("\nEjemplo 2: I 3x3")
print("Matriz:\n", I3)
print("Resultado:", np.linalg.det(I3))

# ============================
# 2. PRODUCTO
# ============================
print("\n=== PROPIEDAD 2: det(AB) = det(A)*det(B) ===")
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
C = np.array([[3,1],[0,2]])
D = np.array([[1,2],[3,4]])

# Ejemplo 1
AB = np.dot(A, B)
print("\nEjemplo 1")
print("A*B =\n", AB)
print("det(AB) =", np.linalg.det(AB))
print("det(A)*det(B) =", np.linalg.det(A)*np.linalg.det(B))

# Ejemplo 2
CD = np.dot(C, D)
print("\nEjemplo 2")
print("C*D =\n", CD)
print("det(C*D) =", np.linalg.det(CD))
print("det(C)*det(D) =", np.linalg.det(C)*np.linalg.det(D))

# ============================
# 3. TRANSPUESTA
# ============================
print("\n=== PROPIEDAD 3: det(A^T) = det(A) ===")
E = np.array([[5,1],[2,3]])
F = np.array([[1,2,3],[0,1,4],[5,6,0]])

# Ejemplo 1
print("\nEjemplo 1")
print("Matriz E:\n", E)
print("det(E) =", np.linalg.det(E))
print("det(E^T) =", np.linalg.det(E.T))

# Ejemplo 2
print("\nEjemplo 2")
print("Matriz F:\n", F)
print("det(F) =", np.linalg.det(F))
print("det(F^T) =", np.linalg.det(F.T))

# ============================
# 4. TRIANGULAR
# ============================
print("\n=== PROPIEDAD 4: det(Triangular) = producto diagonal ===")
T1 = np.array([[2,0],[0,3]])
T2 = np.array([[1,2,0],[0,3,0],[0,0,4]])

print("\nEjemplo 1")
print("Matriz:\n", T1)
print("Producto diagonal 2*3 =", 2*3)
print("Resultado:", np.linalg.det(T1))

print("\nEjemplo 2")
print("Matriz:\n", T2)
print("Producto diagonal 1*3*4 =", 1*3*4)
print("Resultado:", np.linalg.det(T2))

# ============================
# 5. ESCALAR
# ============================
print("\n=== PROPIEDAD 5: det(kA) = k^n * det(A) ===")
A1 = np.array([[1,2],[3,4]])
A2 = np.array([[2,0],[1,3]])

print("\nEjemplo 1: 2*A1")
print("2*A1 =\n", 2*A1)
print("det(2*A1) =", np.linalg.det(2*A1))
print("2^2 * det(A1) =", 4*np.linalg.det(A1))

print("\nEjemplo 2: 3*A2")
print("3*A2 =\n", 3*A2)
print("det(3*A2) =", np.linalg.det(3*A2))
print("3^2 * det(A2) =", 9*np.linalg.det(A2))

# ============================
# 6. INTERCAMBIO DE FILAS
# ============================
print("\n=== PROPIEDAD 6: intercambio de filas cambia signo ===")
M1 = np.array([[1,2],[3,4]])
M2 = np.array([[3,4],[1,2]])
M3 = np.array([[1,0,2],[0,1,3],[4,5,6]])
M4 = np.array([[0,1,3],[1,0,2],[4,5,6]])

print("\nEjemplo 1")
print("Original:\n", M1)
print("Intercambiada:\n", M2)
print("det Original:", np.linalg.det(M1))
print("det Intercambiada:", np.linalg.det(M2))

print("\nEjemplo 2")
print("Original:\n", M3)
print("Intercambiada:\n", M4)
print("det Original:", np.linalg.det(M3))
print("det Intercambiada:", np.linalg.det(M4))

# ============================
# 7. FILA CERO
# ============================
print("\n=== PROPIEDAD 7: fila cero -> det = 0 ===")
Z1 = np.array([[0,0],[1,2]])
Z2 = np.array([[0,0,0],[1,2,3],[4,5,6]])

print("\nEjemplo 1")
print("Matriz:\n", Z1)
print("Resultado:", np.linalg.det(Z1))

print("\nEjemplo 2")
print("Matriz:\n", Z2)
print("Resultado:", np.linalg.det(Z2))

# ============================
# 8. FILAS IGUALES
# ============================
print("\n=== PROPIEDAD 8: filas iguales -> det = 0 ===")
F1 = np.array([[1,2],[1,2]])
F2 = np.array([[3,4,5],[3,4,5],[1,0,2]])

print("\nEjemplo 1")
print("Matriz:\n", F1)
print("Resultado:", np.linalg.det(F1))

print("\nEjemplo 2")
print("Matriz:\n", F2)
print("Resultado:", np.linalg.det(F2))

# ============================
# 9. SUMA DE FILAS
# ============================
print("\n=== PROPIEDAD 9: una fila = suma de otras filas ===")
G1 = np.array([[1,2],[3,4]])
H1 = np.array([[1,2],[4,6]])

G2 = np.array([[1,1,0],[0,1,1],[1,2,1]])
H2 = np.array([[1,1,0],[0,1,1],[1,3,1]])

print("\nEjemplo 1")
print("Original:\n", G1)
print("Modificada:\n", H1)
print("det(G1):", np.linalg.det(G1))
print("det(H1):", np.linalg.det(H1))

print("\nEjemplo 2")
print("Original:\n", G2)
print("Modificada:\n", H2)
print("det(G2):", np.linalg.det(G2))
print("det(H2):", np.linalg.det(H2))
