import numpy as np

from assemble import assemblage

#Paramètres généraux de la simulation

E=210_000
NU=.3
C=E/((1+NU)*(1-2*NU))*np.array([
    [(1-NU), NU, 0],
    [NU,(1-NU), 0],
    [0, 0,(1-2*NU)/2]]
    )
K=assemblage()

# print(calBe(nodes[elements[8159]])) # le code crash à la 8160 ième itération de la boucle au 8159 ième element

    
# Kreack=K
# Freac=F