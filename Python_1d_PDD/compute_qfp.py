import numpy as np

#method: solve linear continuity equations w FDM in the 
#static case for the electron and hole concentrations. Each is
#a second order linear ODE. Use thermal equilibrium and neutral 
#charge properties of ideal contacts to obtain BC's
#see p39 of Bortolossi 2014 for solution

def compute_eqfp(psi, mesh):
    return np.random.rand(1,len(psi))[0]

def compute_hqfp(psi, mesh):
    return np.random.rand(1,len(psi))[0]