import numpy as np
from constants import *
import matplotlib.pyplot as plt
from geometry_mesh_study import geometry_mesh_study

class parameters(object):

    def __init__(self, geometry, V):
        self.N_a = np.zeros((geometry.Ne_1d,))
        self.N_d = np.zeros((geometry.Ne_1d,))
        self.V_bi = kT_q * np.log(N_a * N_d / n_i**2)

        self.dop = np.zeros((geometry.Nn,)) 
        self.psi = np.zeros((geometry.Nn,))
        self.n = np.zeros((geometry.Nn,))
        self.p = np.zeros((geometry.Nn,))

        for i in range(geometry.Ne_1d):
            if geometry.x_ec[i] < geometry.L_n:
                self.N_a[i] = 0
                self.N_d[i] = N_d / n_i
            if geometry.x_ec[i] > geometry.L_n:
                self.N_a[i] = N_a / n_i
                self.N_d[i] = 0

            self.dop[i] = (self.N_d[i] - self.N_a[i])

        x = geometry.x_no.reshape((geometry.Nn,))
        x_half = geometry.L_n
        self.psi = -np.log(N_d/n_i)*np.tanh(200*(x-x_half))

        self.n = np.exp(self.psi)
        self.p = np.exp(-self.psi)

        self.dop[-1] = -N_a / n_i


        
        


