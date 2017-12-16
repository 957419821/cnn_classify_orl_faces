from tools import addNoises
from translationTest import *
import numpy as np
np.lookfor('matlib')

image, vector = initialImage()
result = addNoises(image)
print(np.mat(np.reshape(result[0], (10, 10))))
