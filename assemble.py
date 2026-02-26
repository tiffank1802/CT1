import numpy as np
# from CAL import  C 
import os
from calDet import calDet
from calBe import calBe
# fichiers=[f for f in os.listdir('.') if f.endswith('.txt')]

# Lire les fichiers de NODE et ELEMENT
# nodeData=np.loadtxt("CT/NODE.txt")
# elementData=np.loadtxt("CT/ELEMENT.txt")
nodeData=np.loadtxt("/kaggle/working/CT/NODE.txt")
elementData=np.loadtxt("/kaggle/working/CT/ELEMENT.txt")

#Extraire les coordonnées des noeuds
nodes=nodeData[:,6:8]
ne=nodes.shape[0]
# Extrakre les éléments (triangles)
elements=elementData[:,5:8]
elements=elements.astype(int)
for i,elm in enumerate(elements):
    if elm[0]==elm[1]or elm[0]==elm[2] or elm[2]==elm[1]:
        print(f"{i}")
E=210_000
NU=.3
C=E/((1+NU)*(1-2*NU))*np.array([
    [(1-NU), NU, 0],
    [NU,(1-NU), 0],
    [0, 0,(1-2*NU)/2]]
    )

# Matrice de raideur
K=np.zeros((2*ne,2*ne))
F=np.zeros(2*ne)

def assemblage():
    
    for elm in elements:


        coords=nodes[elm]   
        coords=coords.astype(float)
        Be=calBe(coords)
        Ke=Be.T@C@Be*calDet(coords)/2
        # print(Ke)
        ind0=(elm[0]-1)*2+0
        ind1=(elm[0]-1)*2+1
        ind2=(elm[1]-1)*2+0
        ind3=(elm[1]-1)*2+1
        ind4=(elm[2]-1)*2+0
        ind5=(elm[2]-1)*2+1
        IndiceGlobal=np.array([ind0,ind1,ind2,ind3,ind4,ind5])
        # K[IndiceGlobal,IndiceGlobal]= K[IndiceGlobal,IndiceGlobal]+Ke
        # # print(f"{elm}\n\n{coords}\n{i}")
    return Ke
