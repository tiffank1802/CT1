import numpy as np


def calBe(xy):
    def calDN():
        dN=np.array([
            [-1,1,0],
            [-1,0,1]],dtype=float
        )
        matrice_a_inverser = xy.T @ dN.T
    
        # Vérifier le conditionnement
        cond = np.linalg.cond(matrice_a_inverser)
        seuil = 1e17  # À ajuster selon votre précision
        
        if cond > seuil:
            print(f"Matrice mal conditionnée: cond = {cond}")
            # Option 1: Utiliser la pseudo-inverse
            DN = (np.linalg.pinv(matrice_a_inverser)).T @ dN
            
            # Option 2: Ajouter une petite régularisation
            # epsilon = 1e-6
            # matrice_reg = matrice_a_inverser + epsilon * np.eye(matrice_a_inverser.shape[0])
            # DN = (np.linalg.inv(matrice_reg)).T @ dN
        else:
            # Matrice bien conditionnée, inversion normale
            DN = (np.linalg.inv(matrice_a_inverser)).T @ dN
        return DN
    DN=calDN()
    return  np.array([
            [DN[0,0],0,DN[0,1],0,DN[0,2],0],
            [0,DN[1,0],0,DN[1,1],0,DN[1,2]],
            [ DN[1,0],DN[0,0],DN[1,1],DN[0,1],DN[1,2],DN[0,2]]
            ])

if __name__=="__main__":
    A = np.array([
        [-1, 0],           # c'etait [-1, 1]
        [2, 2],            
        [0, 0]             
    ])
    print(calBe(A))