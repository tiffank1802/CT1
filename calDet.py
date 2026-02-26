import numpy as np

def calDet(xy):
    dN=np.array([
        [-1,1,0],
        [-1,0,1]]
    )
    det=-np.linalg.det(xy.T@dN.T) 
    return det 