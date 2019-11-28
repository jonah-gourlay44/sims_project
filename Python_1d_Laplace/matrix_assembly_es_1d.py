#Electrostatic Solver 1-D
#Matrix Assembly

import numpy as np
from fem_functions import *
import itertools

class matrix_assembly_1d(object):

    def __init__(self, geometry, parameters):
        Nn = geometry.Nn

        self.geometry = geometry
        self.parameters = parameters
        
        self.A = {}
        self.b = np.zeros((Nn,1))

    def build_matrices(self):
        Ne_1d = self.geometry.Ne_1d
        el_1d_no = self.geometry.el_1d_no
        x_no = self.geometry.x_no
        eps_r = self.parameters.eps_r
        rho = self.parameters.rho
        eps_0 = self.parameters.eps_0

        for i in range(Ne_1d):    

            nds_= np.asarray(el_1d_no[i],dtype=np.int)
            xl=x_no[nds_]
            
            ae=dNi_dNj_int_cont_line_Ver_1(xl)
            be=Ni_int_cont_line_Ver_1(xl)

            ae=ae*eps_r[i]
            be=be*rho[i]

            nds_ = tuple(nds_)
            indices = list(itertools.product(nds_,nds_))

            ae_m = ae.shape[0]
            ae_n = ae.shape[1]
            ae = ae.reshape((ae_m*ae_n,))

            for index in indices:
                try:
                    val = self.A[index]
                except KeyError:
                    val = 0
                self.A[index] = val + ae[indices.index(index)]*eps_0
            
            self.b[nds_,0]=self.b[nds_,0]+be.reshape((be.shape[0],))

    def impose_boundary_conditions(self):
        Nn = self.geometry.Nn
        phi_1 = self.parameters.phi_1
        phi_2 = self.parameters.phi_2

        for i in range(Nn):
            index_Nn = (Nn-1, i)
            index_1 = (0, i)

            self.A[index_Nn] = 0; self.A[index_1] = 0

        self.A[(0,0)] = 1; self.b[0] = phi_1
        self.A[(Nn-1, Nn-1)] = 1; self.b[Nn-1] = phi_2
