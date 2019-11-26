import numpy as np
from geometry_mesh import *


#parameters of air
mu_r_a =1
sig_a=0
eps_r_a=1
rho_a=1e-7

#parameters of insulator
mu_r_i=1
sig_i=0
eps_r_i=2
rho_i=0

#boundary conditions (Dirichlet)
phi_1=1.5
phi_2=0

mu_r=np.zeros((Ne_1d,1))
sig=np.zeros((Ne_1d,1))
