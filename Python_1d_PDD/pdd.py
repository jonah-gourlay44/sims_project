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

    def __init__(self, N1, dt, alpha):
        self.V = 0
        self.alpha = alpha
        self.dt = dt

        self.mesh = geometry_mesh_study(N1, 2e-4)
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

        self.d_psi = None
        self.d_2_psi = None
        self.dn_dt = np.zeros((self.mesh.Nn,))
        self.dp_dt = np.zeros((self.mesh.Nn,))

        self.p_prev = np.zeros((self.mesh.Nn,))
        self.n_prev = np.zeros((self.mesh.Nn,))

        self.G = None
    
    def dPsi_dNi_Nj(self, d_psi):
        d_psi = (d_psi[0] + d_psi[1]) / 2

        m = np.zeros((2,2))
        m[0,:] = -1/2
        m[1,:] =  1/2

        m = m * d_psi

        return m

    def dNi_dNj(self, le):
        x_j = le[1][0]; x_i = le[0][0]
        Le = x_j - x_i
        
        m = np.zeros((2,2))
        m[0,:] = 1/Le*np.asarray([1,-1])
        m[1,:] = 1/Le*np.asarray([-1,1])

        return m

    def Ni_Nj(self, le):
        x_j = le[1][0]; x_i = le[0][0]
        Le = x_j - x_i

        m = np.zeros((2,2))
        m[0,0] = -Le/3
        m[0,1] = Le/6
        m[1,0] = Le/6
        m[1,1] = -Le/3

        return m

    def compute_G(self):
        tau_n_norm = tau_n / 1.583e-6
        tau_p_norm = tau_p / 1.583e-6

        self.G = (1 - self.p*self.n) / (tau_p_norm*(self.n + 1) + tau_n_norm*(self.p + 1))

    def build_matrices(self, func):
        Ne_1d = self.mesh.Ne_1d
        el_1d_no = self.mesh.el_1d_no
        Nn = self.mesh.Nn
        x_no = self.mesh.x_no 

        A = {}
        b = {}

        self.d_psi = np.gradient(self.Psi, self.mesh.x_no.reshape((self.mesh.Nn,)))
        self.compute_G()
        
        for i in range(Ne_1d):
            nds_= np.asarray(el_1d_no[i],dtype=np.int)
            xl=x_no[nds_]

            if func == 'psi':
                ae = self.dNi_dNj(xl)
                be = self.Ni_Nj(xl)
            elif func == 'p':
                ae = -1 * self.dPsi_dNi_Nj(self.d_psi[nds_]) * mu_0_p / mu_0 - self.dNi_dNj(xl) * D_0_p / D_0
                be = self.Ni_Nj(xl)
            elif func == 'n':
                ae = self.dPsi_dNi_Nj(self.d_psi[nds_]) * mu_0_n / mu_0 - self.dNi_dNj(xl) * D_0_n / D_0
                be = self.Ni_Nj(xl)
            elif func == 'n_t':
                ae = self.Ni_Nj(xl) - self.dPsi_dNi_Nj(self.d_psi[nds_]) * mu_0_n / mu_0 * self.dt + self.dNi_dNj(xl) * D_0_n / D_0 * self.dt
                be = self.Ni_Nj(xl)
            elif func == 'p_t':
                ae = self.Ni_Nj(xl) + self.dPsi_dNi_Nj(self.d_psi[nds_]) * mu_0_p / mu_0 * self.dt + self.dNi_dNj(xl) * D_0_p / D_0 * self.dt
                be = self.Ni_Nj(xl)

            nds_ = tuple(nds_)
            indices = list(itertools.product(nds_,nds_))

            ae_shape = np.asarray(ae).shape

            ae_m = ae_shape[0]
            ae_n = ae_shape[1]
            ae = ae.reshape((ae_m*ae_n,))

            be_shape = np.asarray(be).shape

            be_m = be_shape[0]
            be_n = be_shape[1]
            be = be.reshape((be_m*be_n,))

            for index in indices:
                try:
                    A_val = A[index]
                except KeyError:
                    A_val = 0
                try:
                    b_val = b[index]
                except KeyError:
                    b_val = 0
                
                A[index] = A_val + ae[indices.index(index)]
                b[index] = b_val + be[indices.index(index)]
        
        Nn = self.mesh.Nn
        A, b = self.apply_bcs(func, A, b, Nn)
                   
        keys, values = zip(*A.items())
        i_A, j_A = zip(*keys)
        i_A = np.array(i_A)
        j_A = np.array(j_A)
        s_A = np.array(values)

        keys, values = zip(*b.items())
        i_b, j_b = zip(*keys)
        i_b = np.array(i_b)
        j_b = np.array(j_b)
        s_b = np.array(values)

        self.K = sparse.csr_matrix((s_A, (i_A, j_A)), shape=(Nn, Nn), dtype=np.float) 
        self.b = sparse.csr_matrix((s_b, (i_b, j_b)), shape=(Nn, Nn), dtype=np.float)

    def apply_bcs(self, func, A, b, Nn):
        for i in range(Nn):
            index_Nn = (Nn-1, i)
            index_1 = (0, i)

            A[index_1] = 0; A[index_Nn] = 0
            b[index_1] = 0; b[index_Nn] = 0
        
        A[(0,0)] = 1; A[(Nn-1, Nn-1)] = 1
        b[(0,0)] = 1; b[(Nn-1, Nn-1)] = 1

        return A, b   

    def solve(self, func):

        if func == 'psi':
            f = self.n - self.p - self.params.dop
            f[0] = 0; f[-1] = self.V - np.log(N_a*N_d/n_i**2)
            b = self.b.dot(f)
            self.Psi = spsolve(self.K, b)
        elif func == 'p':
            f = self.dp_dt + self.G
            f[0] = n_i / N_d; f[-1] = N_a / n_i
            b = self.b.dot(f)
            self.p = self.alpha*spsolve(self.K, b) + (1-self.alpha)*self.p
        elif func == 'n':
            f = self.dn_dt + self.G
            f[0] = N_d / n_i; f[-1] = n_i / N_a
            b = self.b.dot(f)
            self.n = self.alpha*spsolve(self.K, b) + (1-self.alpha)*self.n
        elif func == 'n_t':
            f = self.n + self.G * self.dt
            b = self.b.dot(f)
            self.n = spsolve(self.K, b)
            self.dn_dt = (self.n - f) / self.dt
        elif func == 'p_t':
            f = self.p + self.G * self.dt
            b = self.b.dot(f)
            self.p = spsolve(self.K, b)
            self.dp_dt = (self.p - f) / self.dt
            

    def run_equilibrium(self, num_iter, cutoff):
        iteration = 0; d_psi = 1e3; d_res = 1e3

        while iteration < num_iter and d_psi > cutoff:

            psi = self.Psi

            self.build_matrices('psi')
            self.solve('psi')

            self.build_matrices('n')
            self.solve('n') 
            
            self.build_matrices('p')
            self.solve('p')
            
            self.d_psi = np.gradient(self.Psi, self.mesh.x_no.reshape((self.mesh.Nn,)))
            self.d_2_psi = np.gradient(self.d_psi, self.mesh.x_no.reshape((self.mesh.Nn,)))

            res = self.d_2_psi - (self.n - self.p - self.params.dop)*q
            d_res = np.linalg.norm(res) 
     
            delta = psi - self.Psi

            d_psi = np.linalg.norm(delta)
            iteration += 1

            print('ITERATION: ' + str(iteration) + '\tDELTA: ' + str(d_psi) + '\tRESIDUAL: ' + str(d_res))

    def run_time_step(self):

        self.build_matrices('p_t')
        self.solve('p_t')

        self.build_matrices('n_t')
        self.solve('n_t') 

if __name__ == '__main__':
    f, (ax1, ax2) = plt.subplots(1, 2)

    num_points = 500
    dt = 1e-6
    alpha = 0.01

    my_pdd = pdd(num_points, dt, alpha)

    for i in range(40):
        #ax1.plot(my_pdd.mesh.x_no * my_pdd.mesh.Ldi, my_pdd.p, 'b-', linewidth=1.5, markersize=4)
        ax1.plot(my_pdd.mesh.x_no * my_pdd.mesh.Ldi, my_pdd.n - my_pdd.p - my_pdd.params.dop, 'r-', linewidth=1.5, markersize=4)
        ax2.plot(my_pdd.mesh.x_no * my_pdd.mesh.Ldi, my_pdd.Psi,'b-', linewidth=1.5, markersize=4)
        plt.pause(0.0001)
        ax1.clear()
        ax2.clear()

        for j in range(20):
            my_pdd.run_equilibrium(1000,1e-3)
            my_pdd.run_time_step()

            #ax1.plot(my_pdd.mesh.x_no * my_pdd.mesh.Ldi, my_pdd.p, 'b-', linewidth=1.5, markersize=4)
            ax1.plot(my_pdd.mesh.x_no * my_pdd.mesh.Ldi, my_pdd.n - my_pdd.p - my_pdd.params.dop, 'r-', linewidth=1.5, markersize=4)
            ax2.plot(my_pdd.mesh.x_no * my_pdd.mesh.Ldi, my_pdd.Psi,'b-', linewidth=1.5, markersize=4)
            plt.pause(0.0001)
            ax1.clear()
            ax2.clear()
        
        my_pdd.V += 1

    plt.cla()

    ax2.plot(my_pdd.mesh.x_no * my_pdd.mesh.Ldi, my_pdd.Psi *kT_q)
    ax1.plot(my_pdd.mesh.x_no * my_pdd.mesh.Ldi, my_pdd.p * n_i, label='hole concentration')
    ax1.plot(my_pdd.mesh.x_no * my_pdd.mesh.Ldi, my_pdd.n * n_i, label='electron concentration')

    ax2.set_title('1D Potential in a n-p Junction')
    ax2.set_xlabel('Distance (cm)')
    ax2.set_ylabel('Potential (V)')
    ax1.set_title('1D Hole and Electron Concentrations')
    ax1.set_xlabel('Distance (cm)')
    ax1.set_ylabel('Concentration (cm^-3)')
    ax1.legend()

    plt.show()