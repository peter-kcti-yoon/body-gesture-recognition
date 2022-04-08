
import numpy as np


a =np.ones((10))
print(a.shape)

b = a.reshape(-1,5)
print(b.shape)
print(b.ravel().shape)