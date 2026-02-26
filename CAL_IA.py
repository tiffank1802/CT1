import numpy as np
import matplotlib.pyplot as plt

def calBe(xy):
    

# xy : (3,2)  -> 3 nœuds, 2 coordonnées (x,y)
    dN = np.array([[-1, 1, 0],
                   [-1, 0, 1]], dtype=float)  

# (2,3)

    

# DN = (inv(xy'*dN'))'*dN  en Octave
    

# xy' : (2,3), dN' : (3,2) -> xy'.dot(dN') : (2,2)
    A = xy.T @ dN.T            

# (2,2)
    DN = np.linalg.inv(A).T @ dN   

# (2,3)

    Be = np.array([
        [DN[0, 0], 0.0,       DN[0, 1], 0.0,       DN[0, 2], 0.0],
        [0.0,       DN[1, 0], 0.0,       DN[1, 1], 0.0,       DN[1, 2]],
        [DN[1, 0],  DN[0, 0], DN[1, 1],  DN[0, 1], DN[1, 2],  DN[0, 2]]
    ])
    return Be

def calDet(xy):
    dN = np.array([[-1, 1, 0],
                   [-1, 0, 1]], dtype=float)
    A = xy.T @ dN.T
    Det = -np.linalg.det(A)
    return Det

# --- Programme principal ---

Forces = np.zeros((11, 1))
Deplacements = np.zeros((11, 1))

for j in range(0, 11):   

# j = 0..10

    

# Lecture des fichiers NODE et ELEMENT
    

# NODE.txt : colonnes, on garde les colonnes 7 et 8 (indices 6 et 7 en Python)
    nodeData = np.loadtxt('NODE.txt')
    elementData = np.loadtxt('ELEMENT.txt')

    nodes = nodeData[:, 6:8]      

# (ne,2)
    ne = nodes.shape[0]

    elements = elementData[:, 5:8].astype(int)  

# (nel,3), indices de nœuds (1-based dans le fichier)

    

# Matrice de raideur et vecteur de forces
    K = np.zeros((2*ne, 2*ne))
    F = np.zeros((2*ne, 1))

    

# Matrice de comportement (contrainte plane)
    E = 210000.0
    NU = 0.3
    C = E / ((1 + NU) * (1 - 2*NU)) * np.array([
        [1 - NU, NU,     0.0],
        [NU,     1 - NU, 0.0],
        [0.0,    0.0,    (1 - 2*NU)/2]
    ])

    

# Assemblage
    for e in range(elements.shape[0]):
        

# indices de nœuds de l’élément (1-based dans le fichier)
        n1, n2, n3 = elements[e, :]  

# ex: 1,2,3

        

# coordonnées des 3 nœuds -> (3,2)
        xy = np.vstack((nodes[n1-1, :],
                        nodes[n2-1, :],
                        nodes[n3-1, :]))

        Be = calBe(xy)
        Ke = Be.T @ C @ Be * calDet(xy) / 2.0

        

# indices globaux (0-based pour Python)
        ind1 = (n1 - 1)*2
        ind2 = (n1 - 1)*2 + 1
        ind3 = (n2 - 1)*2
        ind4 = (n2 - 1)*2 + 1
        ind5 = (n3 - 1)*2
        ind6 = (n3 - 1)*2 + 1

        IndiceGlobal = np.array([ind1, ind2, ind3, ind4, ind5, ind6])

        

# Assemblage : K(IndiceGlobal,IndiceGlobal) += Ke
        

# On fait un double parcours pour rester simple
        for a in range(6):
            for b in range(6):
                K[IndiceGlobal[a], IndiceGlobal[b]] += Ke[a, b]

    Kreac = K.copy()
    Freac = F.copy()

    

# Chargement
    P = np.max(K) * 10.0

    for i in range(ne):
        x, y = nodes[i, 0], nodes[i, 1]
        ind1 = 2*i
        ind2 = 2*i + 1

        if (x == 12.5) and (y == 13.75):
            F[ind2, 0] += 1000.0

        if (x == 12.5) and (y == -13.75):
            F[ind2, 0] -= 1000.0

        if (x > 23.75 + j) and (abs(y) < 0.01):
            

# K(ind2,ind2) = K(ind2,ind2) + P;
            K[ind2, ind2] += P

    

# Résolution : U = K^(-1)*F
    U = np.linalg.solve(K, F)

    Reac = Kreac @ U - Freac

    Forces[j, 0] = abs(Reac[1, 0])      

# Reac(2,1) en Octave -> index 1 en Python
    Deplacements[j, 0] = abs(U[1, 0])   

# U(2,1) -> index 1

# Post-traitement
L = np.zeros((10, 1))
G = np.zeros((10, 1))
G_D = np.zeros((10, 1))

for j in range(1, 11):
    G[j-1, 0] = (-Forces[j, 0] + Forces[j-1, 0]) * 0.05
    G_D[j-1, 0] = (Deplacements[j, 0] - Deplacements[j-1, 0]) * 1000.0
    L[j-1, 0] = j - 1

plt.figure()
plt.plot(L, G, '*', label='G (déplacement imposé)')
plt.plot(L, G_D, '-', label='G_D (effort imposé)')
plt.xlabel('L')
plt.ylabel('G')
plt.legend()
plt.grid(True)
plt.savefig("CT/results/solution.png")
