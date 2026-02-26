import pytest
import numpy as np
from assemble import assemblage
from calBe import calBe
from calDet import calDet
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
def test_assemblage():
    A = np.array([
        [-1, 0],           # c'etait [-1, 1]
        [2, 2],            
        [0, 0]             
    ])
    B=assemblage(A)
    C=np.zeros((6,6))
    np.testing.assert_array_almost_equal(B, C, decimal=5)
