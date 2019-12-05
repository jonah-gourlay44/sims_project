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
            Le = xl[1]-xl[0]

            #take linear interpolations of psi, phi_n, phi_v
            psi_lin = lambda x : (self.parameters.psi[nds_[1]] - self.parameters.psi[nds_[0]]) * (x - xl[0]) /Le + self.parameters.psi[nds_[0]]
            phi_n_lin = lambda x : (self.parameters.phi_n[nds_[1]] - self.parameters.phi_n[nds_[0]]) * (x - xl[0]) /Le + self.parameters.phi_n[nds_[0]]
            phi_v_lin = lambda x : (self.parameters.phi_v[nds_[1]] - self.parameters.phi_v[nds_[0]]) * (x - xl[0]) /Le + self.parameters.phi_v[nds_[0]]
            print(psi_lin)
            print(phi_n_lin)
            print(phi_v_lin)
            
            #compute the matrix and vector elements
            ae=dNi_dNj_int_cont_line_Ver_1(xl) + beta_Ni_Nj(xl, psi_lin, phi_n_lin, phi_v_lin)
            be=Ni_f_int_cont_line_Ver_1(xl, self.parameters.psi_pp[i], psi_lin, phi_n_lin, phi_v_lin, self.parameters.n_donor[i] - self.parameters.n_acceptor[i]) #TODO update this fem function 

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

    #TODO check boundary conditions!!!
    def impose_boundary_conditions(self):
        Nn = self.geometry.Nn
        psi_1 = self.parameters.psi_1
        psi_2 = self.parameters.psi_2

        for i in range(Nn):
            index_Nn = (Nn-1, i)
            index_1 = (0, i)

            self.A[index_Nn] = 0; self.A[index_1] = 0

        self.A[(0,0)] = 1; self.b[0] = psi_1
        self.A[(Nn-1, Nn-1)] = 1; self.b[Nn-1] = psi_2
