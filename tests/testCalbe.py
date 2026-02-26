import pytest
import numpy as np



from calBe import calBe
# je dois respecter la structure generale des tests:
# les fichiers de tests sont dans un dossier tests dans le dossier parent qui contient les fichiers à tester
# dans le fichier où on écrit le test, on importe les fichier des fonctions : cet importation se fait comme si les fichiers 
# étaient en fait dans le repertoire courant (tests)
def test_calBe():
    # Création de la matrice A correctement
    A = np.array([
        [-1, 1],           # Première ligne: [-1, 1] au lieu de [1,1] avec A[0,0]=-1
        [2, 2],            # Deuxième ligne: [2, 2] (2*np.ones(2))
        [0, 0]             # Troisième ligne: [0, 0] (np.zeros(2))
    ])
    
    # Calcul avec la fonction à tester
    B = calBe(A)
    
    # Matrice attendue
    C = (1/4) * np.array([
        [-2, 1, 1],
        [2, 1, -3]
    ])
    D=np.array([
            [C[0,0],0,C[0,1],0,C[0,2],0],
            [0,C[1,0],0,C[1,1],0,C[1,2]],
            [ C[1,0],C[0,0],C[1,1],C[0,1],C[1,2],C[0,2]]
            ])
    
    np.testing.assert_array_almost_equal(B, D, decimal=5)
    # conclusion la fonction calBe fonctionne bien 
   