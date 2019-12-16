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
import time

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

        self.delta = np.zeros((self.mesh.Nn,))
        self.V = np.zeros((self.mesh.Nn,))

        self.d_psi = None
        self.d_2_psi = None

    def dNi_dNj_psi(self, le, psi, n, n_c, p_c):
        #le = le *  self.mesh.Ldn / self.mesh.Ldi
        x_j = le[1][0]; x_i = le[0][0]
        Le = (x_j - x_i) #* 1.6583e-7

        ae = np.zeros((2,2))
        ae[:,0] = 1/Le*np.asarray([-1,1])
        ae[:,1] = 1/Le*np.asarray([1,-1])
        #print(ae)
        return ae
    
    def Ni_f_psi(self, le, psi, ind, n, n_c, p_c):
        #le = le *  self.mesh.Ldn / self.mesh.Ldi
        x_j = le[1][0]; x_i = le[0][0]
        Le = (x_j - x_i) 
        
        N_n = self.params.N_d[ind] - self.params.N_a[ind]
        f = (n_c - p_c - N_n) #+ self.mesh.dx_a * self.
        
        f_int = np.linspace(f[0], f[1], n)
        x = np.linspace(x_i,x_j,n).reshape((n,))
        Ni = 1/Le*(x_j -x)
        Nj = 1/Le*(x - x_i)

        be = np.zeros((2,1))
        be[0,0] = integrate.simps(Ni * f_int, x)
        be[1,0] = integrate.simps(Nj * f_int, x)
        #print(be)
        return be 

    def dNi_dNj_Ni_dNj_Ni_Nj(self, le, psi, d_psi, d_2_psi, n, case):
        #le = le *  self.mesh.Ldn / self.mesh.Ldi
        x_j = le[1][0]; x_i = le[0][0]
        Le = x_j - x_i

        switcher = {
            'p': [-D_p, mu_p],
            'n': [D_n, mu_n]
        }

        [D, mu] = switcher[case]

        D = D / self.mesh.Ldi ** 2
        mu = mu / self.mesh.Ldi ** 2 * kT_q

        psi = np.linspace(psi[0], psi[1], n)
        x = np.linspace(x_i, x_j, n)
        d_psi = np.linspace(d_psi[0], d_psi[1], n)
        d_2_psi = np.linspace(d_2_psi[0], d_2_psi[1], n)
        dNi = -1/Le; dNj = 1/Le
        Ni = 1/Le*(x_j - x); Nj = 1/Le*(x-x_i)

        ae = np.zeros((2,2))
        ae[0,0] = integrate.simps((D*dNi**2  - mu*d_psi*Ni*dNi - mu*d_2_psi*Ni**2), x)
        ae[0,1] = integrate.simps((D*dNi*dNj - mu*d_psi*Ni*dNj - mu*d_2_psi*Ni*Nj), x)
        ae[1,0] = integrate.simps((D*dNj*dNi - mu*d_psi*Nj*dNi - mu*d_2_psi*Nj*Ni), x)
        ae[1,1] = integrate.simps((D*dNj**2  - mu*d_psi*Nj*dNj - mu*d_2_psi*Nj**2), x)

        return ae

    def build_matrices(self, case, func):
        n = 1000
        Ne_1d = self.mesh.Ne_1d
        el_1d_no = self.mesh.el_1d_no
        Nn = self.mesh.Nn
        x_no = self.mesh.x_no 

        A = {}
        b = np.zeros((Nn,1))

        self.d_psi = np.gradient(self.Psi, self.mesh.dx_a)
        self.d_2_psi = np.gradient(self.d_psi, self.mesh.dx_a)
        
        for i in range(Ne_1d):
            nds_= np.asarray(el_1d_no[i],dtype=np.int)
            xl=x_no[nds_]

            if func == 'psi':
                ae = self.dNi_dNj_psi(xl, self.Psi[nds_], n, self.n[nds_], self.p[nds_])
                be = self.Ni_f_psi(xl, self.Psi[nds_], i, n, self.n[nds_], self.p[nds_])
            else:
                ae = self.dNi_dNj_Ni_dNj_Ni_Nj(xl, self.Psi[nds_], self.d_psi[nds_], self.d_2_psi[nds_], n, func)
                be = np.zeros((2,1))

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

                    A[index_1] = 0; A[index_Nn] = 0
                
                A[(0,0)] = 1; b[0] = 0
                A[(Nn-1, Nn-1)] = 1; b[Nn-1] = 0
            elif case == 'eq':
                if func == 'psi':
                    for i in range(Nn):
                        index_1 = (0, i)

                        A[index_1] = 0
                
                    A[(0,0)] = 1; b[0] = np.log(N_d*N_a/n_i**2) 
                    A[(Nn-1, Nn-1)] = 0 
                else:
                    for i in range(Nn):
                        index_1 = (0, i)
                        index_Nn = (Nn-1, i)
                    
                        A[index_1] = 0; A[index_Nn] = 0

                    c_end = 1
                    c_start =1
                    A[(0,0)] = 1; A[(Nn-1, Nn-1)] = 1
                    if func == 'p':
                        c_start = (n_i ** 2/ N_d) / N
                        c_end = N_a / N
                    elif func == 'n':
                        c_start = N_d  / N
                        c_end = (n_i ** 2 / N_a) / N
                    
                    b[0] = c_start; b[-1] = c_end
                   
            
            keys, values = zip(*A.items())
            i, j = zip(*keys)
            i = np.array(i)
            j = np.array(j)
            s = np.array(values)

            self.K = sparse.csr_matrix((s, (i, j)), shape=(Nn, Nn), dtype=np.float) 
            #print(self.K.toarray())
            self.b = b
            #print(self.b)

    def solve(self, case, func):
        
        if case =='eq':
            if func == 'psi':
                self.Psi = spsolve(self.K, self.b)
            elif func == 'p':
                self.p = spsolve(self.K, self.b) 
            elif func == 'n':
                self.n = spsolve(self.K, self.b) 
            
        elif case=='non_eq':
            self.delta = solve(self.K, self.b)


    def run_equilibrium(self, num_iter, cutoff):
        iteration = 0; d_psi = 1e3

        plt.ion()

        f, (ax1, ax2) = plt.subplots(1, 2)

        while iteration < num_iter and d_psi > cutoff:
             
            ax1.plot(self.mesh.x_no * self.mesh.Ldn, self.p * N / n_i, 'b-', linewidth=1.5, markersize=4)
            ax1.plot(self.mesh.x_no * self.mesh.Ldn, self.n * N / n_i, 'r-', linewidth=1.5, markersize=4)
            ax2.plot(self.mesh.x_no * self.mesh.Ldn, self.Psi * kT_q,  'b-', linewidth=1.5, markersize=4)
            plt.pause(0.0001)
            ax1.clear()
            ax2.clear()

            psi = self.Psi

            self.build_matrices('eq', 'psi')
            self.solve('eq', 'psi')

            self.build_matrices('eq', 'n')
            self.solve('eq', 'n') 
            
            self.build_matrices('eq', 'p')
            self.solve('eq', 'p')

            #self.p = np.exp(-1*self.Psi)
            #self.n = np.exp(self.Psi)

            #res = 
     
            delta = psi - self.Psi

            d_psi = np.linalg.norm(delta)
            iteration += 1

            print('ITERATION: ' + str(iteration) + ' DELTA: ' + str(d_psi))
        

    def run_nonequilibrium(self, num_iter, cutoff):

        iteration = 0; d_psi = 1e3

        while iteration < num_iter:
            self.build_matrices('non_eq', 'psi')
            psi = self.Psi
            self.solve('non_eq', 'psi')
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
    pdd = pdd(100,0.0)
    pdd.run_equilibrium(101,1e-5)
    print('BUILT IN VOLTAGE: ' + str((np.max(pdd.Psi) - np.min(pdd.Psi)) * kT_q) + ' V')
    plt.cla()
    plt.ioff()

    #pdd.run_nonequilibrium(100,1e-5,)

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax2.plot(pdd.mesh.x_no * pdd.mesh.Ldi, pdd.Psi *kT_q)
    ax1.plot(pdd.mesh.x_no * pdd.mesh.Ldi, pdd.p * N)
    ax1.plot(pdd.mesh.x_no * pdd.mesh.Ldi, pdd.n * N)

    plt.show()

