import numpy as np
from scipy import integrate
from geometry_mesh_study import geometry_mesh_study
from constants import *
import matplotlib.pyplot as plt

N = 100
mesh = geometry_mesh_study(N,0.02)
mesh.discretize()

x = mesh.x_no

y = np.tanh(-x + 3)* np.log(N_a*N_d/n_i**2)

plt.plot(x,y)
plt.show()
