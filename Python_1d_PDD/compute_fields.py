#Electrostatic Solver 1-D
#Compute Fields

import numpy as np
from fem_functions import dNi_line_Ver_1

class field_computation(object):

    def __init__(self, geometry, parameters, Phi):
        Nn = geometry.Nn

        self.geometry = geometry
        self.parameters = parameters
        self.Phi = Phi

        self.Ex = np.zeros((Nn, 1))     #Electric field
        self.Dx = np.zeros((Nn, 1))     #Electric flux density
        self.supp = np.zeros((Nn, 1))   #Nodal supports
        self.We1 = 0
        self.We2 = 0
        self.We3 = 0

    def compute_fields(self):
        el_1d_no = self.geometry.el_1d_no
        x_no = self.geometry.x_no
        Ne_1d = self.geometry.Ne_1d
        eps_0 = self.parameters.eps_0
        eps_r = self.parameters.eps_r
        rho = self.parameters.rho
        Nn = self.geometry.Nn

        for i in range(Ne_1d):

            nds_ = np.asarray(el_1d_no[i], dtype=np.int)
            xl = x_no[nds_]
            Le = xl[1] - xl[0]
            pot = self.Phi[nds_]

            ex_e = dNi_line_Ver_1(xl)
            dx_e = ex_e * eps_r[i] * eps_0

            self.Ex[nds_[0]] = self.Ex[nds_[0]] - (ex_e[0] * pot[0] + ex_e[1] * pot[1]) * Le
            self.supp[nds_[0]] = self.supp[nds_[0]] + Le
            self.Ex[nds_[1]] = self.Ex[nds_[1]] - (ex_e[0] * pot[0] + ex_e[1] * pot[1]) * Le
            self.supp[nds_[1]] = self.supp[nds_[1]] + Le

            self.Dx[nds_[0]] = self.Dx[nds_[0]] - (dx_e[0] * pot[0] + dx_e[1] * pot[1]) * Le
            self.Dx[nds_[1]] = self.Dx[nds_[1]] - (dx_e[0] * pot[0] + dx_e[1] * pot[1]) * Le

            Exe = (ex_e[0] * pot[0] + ex_e[1] * pot[1])
            Dxe = (dx_e[0] * pot[0] + dx_e[1] * pot[1])

            self.We1 = self.We1 + Exe * Dxe * Le
            self.We2 = self.We2 + rho[i] * np.mean(pot) * Le
        
        for i in range(Nn):

            self.Ex[i] = self.Ex[i] / self.supp[i]
            self.Dx[i] = self.Dx[i] / self.supp[i]

        self.We1 = 0.5 * self.We1
        self.We2 = 0.5 * self.We2
        self.We3 = 0.5 * self.Dx[0] * self.Phi[0]


