import numpy as np
from scipy import integrate
from geometry_mesh_study import geometry_mesh_study

N = 100
mesh = geometry_mesh_study(N)
mesh.discretize()

x = mesh.x_no

y = np.sin(x)

Ne_1d = mesh.Ne_1d

integral = 0

for i in range(Ne_1d):
    nds_= np.asarray(mesh.el_1d_no[i],dtype=np.int)
    xl=x[nds_].reshape((2,))
    yl=y[nds_].reshape((2,))
    
    x_p = np.linspace(xl[0],xl[1],100)
    y_p = np.linspace(yl[0],yl[1],100)
    
    integral += integrate.simps(y_p, x_p)
    print(integral)


print(integral)