import numpy as np
import itertools
from geometry_mesh_study import geometry_mesh_study
from model_parameters import parameters
from electrical_analysis_rough import compute_qfp
from scipy import integrate
from scipy import sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve
import matplotlib.pyplot as plt
from constants import *

class pdd(object):

    def __init__(self, N1, V):
        W = np.sqrt(2*eps*(N_a + N_d)*V_bi/(q_Na*N_d))
        Wn = W*np.sqrt(N_a/(N_a+N_d))
        Wp = W*np.sqrt(N_d/(N_a+N_d))
        Wone = np.sqrt(2*eps*V_bi/(q_Na))
        E_p = q_Nd * Wn / eps

        L = 0
        if L < Wn:
            L = Wn
        if L < Wp:
            L = Wp
        L = 20*L

        self.V = V

        self.mesh = geometry_mesh_study(N1, 0.02)
        self.mesh.discretize()
        self.params = parameters(self.mesh, self.V)
        self.analysis = compute_qfp(self.mesh, self.V)

        self.Psi = self.params.psi 
        self.Phi_n = self.analysis.Phi_p 
        self.p = self.params.p
        self.n = self.params.n
        self.Phi_p = self.analysis.Phi_n 

        self.K = None
        self.b = None

        self.delta = np.zeros((self.mesh.Nn,))
        self.V = np.zeros((self.mesh.Nn,))

    def dNi_dNj_beta_Ni_Nj(self, le, psi, n, n_c, p_c):
        x_j = le[1]; x_i = le[0]
        Le = (x_j - x_i) 

        ae = np.zeros((2,2))
        ae[:,0] = 1/Le*np.asarray([1,-1])
        ae[:,1] = 1/Le*np.asarray([-1,1])

        return ae
    
    def Ni_f(self, le, psi, ind, n, n_c, p_c):
        x_j = le[1]; x_i = le[0]
        Le = (x_j - x_i) 
        
        N_c = self.params.N_d[ind] - self.params.N_a[ind]
        f = (n_c - p_c - N_c) * n_i / N
        
        be = np.zeros((2,1))
        be[:,0] = Le/2 * f
        print(be)
        return be 

    def build_matrices_eq(self):
        Nn = self.mesh.Nn
        dx = self.mesh.dx_a
        Na = self.params.N_a
        Nd = self.params.N_d

        matrix = np.zeros((Nn, Nn))
        column = np.zeros((Nn,))

        for i in range(1,Nn-1):
            row = np.asarray([1, -(2 + n_i/N*dx**2*(np.exp(self.Psi[i]) - np.exp(-1 * self.Psi[i]))), 1])
            matrix[i,i-1:i+2] = row
            entry = -self.Psi[i-1] + 2*self.Psi[i] - self.Psi[i+1] + n_i/N*dx**2*(np.exp(self.Psi[i])-np.exp(-1*self.Psi[i]) + (Na[i] - Nd[i]) / n_i)
            column[i] = entry

        #matrix[0,0:3] = np.asarray([(1 + n_i/N*dx**2*(np.exp(0.3495) - np.exp(-1 * 0.3495))), -2, 1])
        #matrix[-1,-3:] = np.asarray([1, -2, (1 + n_i/N*dx**2*(np.exp(self.Psi[-1]) - np.exp(-1*self.Psi[-1])))])
        #column[0] = 0#-1 * self.Psi[2] + 2*self.Psi[1] - 0.3495 + n_i/N*(np.exp(0.3495) - np.exp(-0.3495) + (Na[0]-Nd[0])/n_i)
        #column[-1] = -1*self.Psi[-1] + 2*self.Psi[-2] -self.Psi[-3] + n_i/N*(np.exp(self.Psi[-1]) - np.exp(-1*self.Psi[-1]) + (Na[-1] - Nd[-1]) / n_i)

        self.K = matrix
        self.b = column


    def build_matrices(self, case):
        n = 1000
        Ne_1d = self.mesh.Ne_1d
        el_1d_no = self.mesh.el_1d_no
        Nn = self.mesh.Nn
        x_no = self.mesh.x_no 

        A = {}
        b = np.zeros((Nn,1))

        for i in range(Ne_1d):
            nds_= np.asarray(el_1d_no[i],dtype=np.int)
            xl=x_no[nds_]

            ae = self.dNi_dNj_beta_Ni_Nj(xl, self.Psi[nds_], n, self.n[nds_], self.p[nds_])
            be = self.Ni_f(xl, self.Psi[nds_], i, n, self.n[nds_], self.p[nds_])

            nds_ = tuple(nds_)
            indices = list(itertools.product(nds_,nds_))

            ae_shape = np.asarray(ae).shape

            ae_m = ae_shape[0]
            ae_n = ae_shape[1]
            ae = ae.reshape((ae_m*ae_n,))

            for index in indices:
                try:
                    val = A[index]
                except KeyError:
                    val = 0
                A[index] = val + ae[indices.index(index)]
            
            be_shape = np.asarray(be).shape

            b[nds_,0] = b[nds_,0]+be.reshape((be_shape[0],))
            
            Nn = self.mesh.Nn

            if case == 'non_eq':
                for i in range(Nn):
                    index_Nn = (Nn-1, i)
                    index_1 = (0, i)

                    A[index_Nn] = 0; A[index_1] = 0
                
                A[(0,0)] = 1; b[0] = 0
                A[(Nn-1, Nn-1)] = 1; b[Nn-1] = 0
            elif case == 'eq':
                for i in range(Nn):
                    index_1 = (0, i)

                    A[index_1] = 0
                
                A[(0,0)] = 1; b[0] = 0
            
            keys, values = zip(*A.items())
            i, j = zip(*keys)
            i = np.array(i)
            j = np.array(j)
            s = np.array(values)

            self.K = sparse.csr_matrix((s, (i, j)), shape=(Nn, Nn), dtype=np.float)
            self.b = b

    def solve(self, case):
        
        if case=='non_eq':
            self.Psi = spsolve(self.K, self.b)
        elif case=='eq':
            self.delta = solve(self.K, self.b)


    def run_equilibrium(self, num_iter, cutoff):
        iteration = 0; d_psi = 1e3
        while iteration < num_iter and d_psi > cutoff: 
            #self.build_matrices('eq')
            #psi = self.Psi
            self.build_matrices_eq()
            self.solve('eq')

            #delta = psi - self.Psi
            self.Psi = self.Psi + self.delta
            #self.p = np.exp(self.Phi_p-self.Psi) 
            #self.n = np.exp(self.Psi-self.Phi_n) 
            print(self.K)
            d_psi = np.linalg.norm(self.delta)
            iteration += 1

            print('ITERATION: ' + str(iteration) + ' DELTA: ' + str(d_psi))

    def run_nonequilibrium(self, num_iter, cutoff):

        iteration = 0; d_psi = 1e3

        while iteration < num_iter:
            self.build_matrices('non_eq')
            psi = self.Psi
            self.solve('non_eq')
            delta = psi - self.Psi
            self.analysis.integrate(self.Psi, self.params)
            self.Phi_n = self.analysis.Phi_n
            self.Phi_p = self.analysis.Phi_p

            self.p = np.exp(self.Phi_p-self.Psi)
            self.n = np.exp(self.Psi-self.Phi_n)

            d_psi = np.linalg.norm(delta)
            iteration += 1

            print('ITERATION: ' + str(iteration) + ' DELTA: ' + str(d_psi))

if __name__ == '__main__':
    pdd = pdd(1000,0.0)
    #plt.plot(pdd.mesh.x_no, pdd.p)
    #plt.plot(pdd.mesh.x_no, pdd.Psi)
    #plt.show()
    pdd.run_equilibrium(1000,1e-5)
    print((np.max(pdd.Psi) - np.min(pdd.Psi))*kT_q)
    plt.plot(pdd.mesh.x_no * L_D, pdd.Psi * kT_q)
    #plt.plot(pdd.mesh.x_no * pdd.mesh.Ldi, pdd.p)
    #plt.plot(pdd.mesh.x_no * pdd.mesh.Ldi, pdd.n)

    plt.show()