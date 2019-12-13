import numpy as np
import itertools
from geometry_mesh_study import geometry_mesh_study
from model_parameters import parameters
from electrical_analysis_rough import compute_qfp
from scipy import integrate
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from constants import *

class pdd(object):

    def __init__(self, N1, V):
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

    def dNi_dNj_beta_Ni_Nj(self, le, psi, n, n_c, p_c):
        x_j = le[1]; x_i = le[0]
        Le = x_j - x_i

        #x = np.linspace(x_i, x_j, n).reshape((n,))
        #psi = np.linspace(psi[0], psi[1], n) 
        #n_c = np.linspace(p_c[0], n_c[1], n)
        #p_c = np.linspace(p_c[0], p_c[1], n)

        #beta = -(n_c + p_c)
        #dNi = -1/Le; dNj = 1/Le
        #Ni = 1/Le * (x_j - x); Nj = 1/Le * (x - x_i)

        ae = np.zeros((2,2))
        ae[:,0] = 1/Le*np.asarray([1,-1])
        ae[:,1] = 1/Le*np.asarray([-1,1])
        #ae[0,0] = integrate.simps(dNi ** 2 + beta * Ni ** 2, x)
        #ae[0,1] = integrate.simps(dNi * dNj + beta * Ni * Nj, x)
        #ae[1,0] = ae[0,1]
        #ae[1,1] = integrate.simps(dNj ** 2 + beta * Nj ** 2, x)

        return ae
    
    def Ni_f(self, le, psi, ind, n, n_c, p_c):
        x_j = le[1]; x_i = le[0]
        Le = x_j - x_i
        
        #x = np.linspace(x_i, x_j, n).reshape((n,))
        #psi = np.linspace(psi[0], psi[1], n)
        #n_c = np.linspace(n_c[0], n_c[1], n)
        #p_c = np.linspace(p_c[0], p_c[1], n)

        #d_2_psi = np.gradient(np.gradient(psi))
        
        N_c = self.params.N_d[ind] - self.params.N_a[ind]
        #N_c = (np.ones((n,)) * n_c) / n_i
        
        #f = -d_2_psi + n_c - p_c - N_c 
        f = (n_c - p_c - N_c) * n_i / N
        
        #Ni = 1/Le * (x_j - x); Nj = 1/Le * (x - x_i)

        be = np.zeros((2,1))
        be[:,0] = Le/2 * f
        #be[0,0] = integrate.simps(np.multiply(Ni, f), x)
        #be[1,0] = integrate.simps(np.multiply(Nj, f), x)

        return be 

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
                    #index_Nn = (Nn-1, i)
                    index_1 = (0, i)

                    #A[index_Nn] = 0
                    A[index_1] = 0
                
                #A[(Nn-1, Nn-1)] = 1; b[-1] = 0
                A[(0,0)] = 1; b[0] = 0
            
            keys, values = zip(*A.items())
            i, j = zip(*keys)
            i = np.array(i)
            j = np.array(j)
            s = np.array(values)

            self.K = sparse.csr_matrix((s, (i, j)), shape=(Nn, Nn), dtype=np.float)
            self.b = b

    def solve(self):

        #delta = spsolve(self.K, self.b)
        self.Psi = spsolve(self.K, self.b)
        #return delta

    def run_equilibrium(self, num_iter, cutoff):
        iteration = 0; d_psi = 1e3
        
        while iteration < num_iter: 
            self.build_matrices('eq')
            psi = self.Psi

            self.solve()

            delta = psi - self.Psi
            self.p = np.exp(-1 * self.Psi) 
            self.n = np.exp(self.Psi) 
            #print(np.exp(self.Psi))
            #self.p = self.p / np.linalg.norm(self.p)
            #self.n = self.n / np.linalg.norm(self.n)
            #zz_1 = 0.5 * (self.params.N_d[0] - self.params.N_a[0])
            #zz_2 = 0.5 * (self.params.N_d[-1] - self.params.N_a[-1])
            #xx_1 = zz_1 * (1 + np.sqrt(1 + 1/zz_1**2))
            #xx_2 = zz_2 * (1 - np.sqrt(1 + 1/zz_2**2))
            #self.p[0] = 1/xx_1 * n_i / N
            #self.n[0] = xx_1 * n_i / N
            #self.p[-1] = 1/xx_2 * n_i / N
            #self.n[-1] = xx_2 * n_i / N

            d_psi = np.linalg.norm(delta)
            iteration += 1

            print('ITERATION: ' + str(iteration) + ' DELTA: ' + str(d_psi))

    def run_nonequilibrium(self, num_iter, cutoff):

        iteration = 0; d_psi = 1e3

        while iteration < num_iter:
            self.build_matrices('non_eq')
            psi = self.Psi
            #delta = self.solve()
            self.solve()
            #self.Psi = self.Psi + delta
            delta = psi - self.Psi
            self.analysis.integrate(self.Psi, self.params)
            self.Phi_n = self.analysis.Phi_n
            self.Phi_p = self.analysis.Phi_p

            self.p = np.exp(self.Phi_p-self.Psi)
            self.n = np.exp(self.Psi-self.Phi_n)
            #self.p[0] = n_i / N_d 
            #self.n[0] = N_d / n_i
            #self.p[-1] = N_a / n_i
            #self.n[-1] = n_i / N_a

            d_psi = np.linalg.norm(delta)
            iteration += 1

            print('ITERATION: ' + str(iteration) + ' DELTA: ' + str(d_psi))

if __name__ == '__main__':
    pdd = pdd(200,0.0)
    
    pdd.run_equilibrium(5,1e-3)
    print((np.max(pdd.Psi) - np.min(pdd.Psi))*kT_q)
    #plt.plot(pdd.mesh.x_no * L_D, pdd.Psi / kT_q)
    plt.plot(pdd.mesh.x_no * pdd.mesh.Ldi, pdd.p)
    plt.plot(pdd.mesh.x_no * pdd.mesh.Ldi, pdd.n)

    plt.show()