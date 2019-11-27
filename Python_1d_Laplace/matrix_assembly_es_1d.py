#Electrostatic Solver 1-D
#Matrix Assembly

from scipy import sparse
import numpy as np
from fem_functions import *

class matrix_assembly_1d(object):

    def __init__(self, geometry, parameters):
        Nn = geometry.Nn
        Ne_1d = geometry.Ne_1d
        el_1d_no = geometry.el_1d_no
        x_no = geometry.x_no
        eps_r = parameters.eps_r
        rho = parameters.rho
        eps_0 = parameters.eps_0
        
        self.A = sparse.bsr_matrix((Nn, Nn), dtype=np.float).toarray()
        self.b = np.zeros((Nn,1))
        self.X = np.zeros((Nn,1))

        for i in range(Ne_1d):

            nds_= np.asarray(el_1d_no[i],dtype=np.int)
            xl=x_no[nds_]
            
            ae=dNi_dNj_int_cont_line_Ver_1(xl)
            be=Ni_int_cont_line_Ver_1(xl)

            ae=ae*eps_r[i]
            be=be*rho[i]
            print(self.A[nds_][nds_])
            self.A[nds_][nds_]=self.A[nds_][nds_]+ae
            self.b[nds_,1]=self.b[nds_,1]+be

        self.A=self.A*eps_0