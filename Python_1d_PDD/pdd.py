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
        self.alpha = 0.05
        self.dt = 1e-7

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

    def dNi_dNj_psi(self, le, psi, n, n_c, p_c):
        x_j = le[1][0]; x_i = le[0][0]
        Le = (x_j - x_i)

        ae = np.zeros((2,2))
        ae[:,0] = 1/Le*np.asarray([-1,1])
        ae[:,1] = 1/Le*np.asarray([1,-1])

        return ae
    
    def Ni_f_psi(self, le, psi, ind, n, n_c, p_c):
        x_j = le[1][0]; x_i = le[0][0]
        Le = (x_j - x_i) 
        
        N_n = self.params.dop[ind]
        f = (n_c - p_c - N_n) 
        
        x = np.linspace(x_i,x_j,n).reshape((n,))
        Ni = 1/Le*(x_j - x)
        Nj = 1/Le*(x - x_i)
        
        be = np.zeros((2,1))
        be[:,0] = integrate.simps(Ni, x) * f

        return be 

    def dNi_dNj_Ni_dNj_Ni_Nj(self, le, psi, d_psi, n, case):
        x_j = le[1][0]; x_i = le[0][0]
        Le = x_j - x_i

        switcher = {
            'p': [D_p, mu_p],
            'n': [D_n, mu_n]
        }

        [D, mu] = switcher[case]

        D = D / D_0
        mu = mu / mu_0

        psi = np.linspace(psi[0], psi[1], n)
        x = np.linspace(x_i, x_j, n)
        d_psi = np.linspace(d_psi[0], d_psi[1], n)

        dNi = -1/Le; dNj = 1/Le
        Ni = 1/Le*(x_j - x); Nj = 1/Le*(x - x_i)

        ae = np.zeros((2,2))
        if case == 'n':
            ae[0,0] = integrate.simps((D*dNi**2  + mu*d_psi*Ni*dNi), x) 
            ae[0,1] = integrate.simps((D*dNi*dNj + mu*d_psi*Ni*dNj), x) 
            ae[1,0] = integrate.simps((D*dNj*dNi + mu*d_psi*Nj*dNi), x) 
            ae[1,1] = integrate.simps((D*dNj**2  + mu*d_psi*Nj*dNj), x) 
        elif case == 'p':
            ae[0,0] = integrate.simps((D*dNi**2  - mu*d_psi*Ni*dNi), x) 
            ae[0,1] = integrate.simps((D*dNi*dNj - mu*d_psi*Ni*dNj), x) 
            ae[1,0] = integrate.simps((D*dNj*dNi - mu*d_psi*Nj*dNi), x) 
            ae[1,1] = integrate.simps((D*dNj**2  - mu*d_psi*Nj*dNj), x) 
        
        return ae

    def Ni_Nj_G(self, le, n_c, p_c, ind, n, case):
        x_j = le[1][0]; x_i = le[0][0]
        Le = x_j - x_i
        
        x = np.linspace(x_i, x_j, n)
        Ni = 1/Le*(x_j - x); Nj = 1/Le*(x - x_i)

        G = (1-n_c*p_c*n_i**2) / (tau_n*(p_c*n_i+1) + tau_p*(n_c*n_i+1)) / 5.657e15

        if case == 'p':
            G = -G
            d_dt = self.dp_dt[ind]
        elif case == 'n':
            d_dt = self.dn_dt[ind]

        be = np.zeros((2,1))
        be[:,0] = integrate.simps(Ni*Nj,x) * (G - d_dt)

        return be

    def time_step(self, le, n_c, p_c, psi, d_psi, n, case):
        dt = self.dt
        x_j = le[1][0]; x_i = le[0][0]
        Le = x_j - x_i

        switcher = {
            'p': [D_p, mu_p],
            'n': [D_n, mu_n]
        }

        [D, mu] = switcher[case]

        D = D / D_0
        mu = mu / mu_0

        x = np.linspace(x_i, x_j, n)
        Ni = 1/Le*(x_j - x); Nj = 1/Le*(x - x_i)
        dNi = -1/Le; dNj = 1/Le

        G = (1-n_c*p_c*n_i**2) / (tau_n*(p_c*n_i+1) + tau_p*(n_c*n_i+1)) / 5.657e15

        if case == 'p':
            G = -G
            c = p_c
        elif case == 'n':
            c = n_c

        n_c = np.linspace(n_c[0], n_c[1], n)
        p_c = np.linspace(p_c[0], p_c[1], n)
        psi = np.linspace(psi[0], psi[1], n)
        d_psi = np.linspace(d_psi[0], d_psi[1], n)

        ae = np.zeros((2,2))
        ae[0,0] = integrate.simps(Ni**2 + mu*d_psi*Ni*dNi*dt + D*dNi**2*dt, x)
        ae[0,1] = integrate.simps(Ni*Nj + mu*d_psi*Ni*dNj*dt + D*dNi*dNj*dt, x)
        ae[1,0] = integrate.simps(Nj*Ni + mu*d_psi*Nj*dNi*dt + D*dNj*dNi*dt, x)
        ae[1,1] = integrate.simps(Nj**2 + mu*d_psi*Nj*dNj*dt + D*dNj**2*dt, x)

        be = np.zeros((2,1))
        be[:,0] = integrate.simps(Ni*Nj, x) * (c + G * dt)

        return ae, be

    def build_matrices(self, case, func):
        n = 1000
        Ne_1d = self.mesh.Ne_1d
        el_1d_no = self.mesh.el_1d_no
        Nn = self.mesh.Nn
        x_no = self.mesh.x_no 

        A = {}
        b = np.zeros((Nn,1))

        self.d_psi = np.gradient(self.Psi, self.mesh.x_no.reshape((self.mesh.Nn,)))
        self.d_psi[0] = 0; self.d_psi[-1] = 0
        self.d_2_psi = np.gradient(self.d_psi, self.mesh.x_no.reshape((self.mesh.Nn,)))
        
        for i in range(Ne_1d):
            nds_= np.asarray(el_1d_no[i],dtype=np.int)
            xl=x_no[nds_]

            if func == 'psi':
                ae = self.dNi_dNj_psi(xl, self.Psi[nds_], n, self.n[nds_], self.p[nds_])
                be = self.Ni_f_psi(xl, self.Psi[nds_], nds_, n, self.n[nds_], self.p[nds_])
            elif case == 'non_eq':
                ae, be = self.time_step(xl, self.n[nds_], self.p[nds_], self.Psi[nds_], self.d_psi[nds_], n, func)
            else:
                ae = self.dNi_dNj_Ni_dNj_Ni_Nj(xl, self.Psi[nds_], self.d_psi[nds_], n, func)
                be = self.Ni_Nj_G(xl, self.n[nds_], self.p[nds_], nds_, n, func)

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

            if func == 'p':
                start = (n_i ** 2/ N_d) / n_i
                end = N_a / n_i

            elif func == 'n':
                start = N_d  / n_i
                end = (n_i ** 2 / N_a) / n_i

            A[(0,0)] = 1; b[0] = start
            A[(Nn-1, Nn-1)] = 1; b[-1] = end

        elif case == 'eq':
            for i in range(Nn):
                index_1 = (0, i)
                index_Nn = (Nn-1, i)

                A[index_1] = 0; A[index_Nn] = 0

            if func == 'psi':
                start = 0
                end = self.V-1*np.log(N_a*N_d/n_i**2)
                A[(1,0)] = 0
                    
            elif func == 'p':
                start = (n_i ** 2/ N_d) / n_i
                end = N_a / n_i

            elif func == 'n':
                start = N_d  / n_i
                end = (n_i ** 2 / N_a) / n_i
                
            A[(Nn-1,Nn-1)] = 1; b[-1] = end
            A[(0,0)] = 1; b[0] = start
                   
        keys, values = zip(*A.items())
        i, j = zip(*keys)
        i = np.array(i)
        j = np.array(j)
        s = np.array(values)

        self.K = sparse.csr_matrix((s, (i, j)), shape=(Nn, Nn), dtype=np.float) 
        self.b = b.reshape((Nn,))

    def solve(self, case, func):
        
        if case =='eq':
            if func == 'psi':
                self.Psi = spsolve(self.K, self.b) 
            elif func == 'p':
                self.p = self.alpha*spsolve(self.K, self.b) + (1-self.alpha)*self.p
            elif func == 'n':
                self.n = self.alpha*spsolve(self.K, self.b) + (1-self.alpha)*self.n
            
        elif case=='non_eq':
            if func == 'p':
                self.p = spsolve(self.K, self.b)
            elif func == 'n':
                self.n = spsolve(self.K, self.b)

    def run_equilibrium(self, num_iter, cutoff):
        iteration = 0; d_psi = 1e3; d_res = 1e3

        plt.ion()

        f, (ax1, ax2) = plt.subplots(1, 2)

        while iteration < num_iter and d_psi > cutoff:
             
            ax1.plot(self.mesh.x_no * self.mesh.Ldi, self.p, 'b-', linewidth=1.5, markersize=4)
            ax1.plot(self.mesh.x_no * self.mesh.Ldi, self.n, 'r-', linewidth=1.5, markersize=4)
            ax2.plot(self.mesh.x_no * self.mesh.Ldi, self.Psi,'b-', linewidth=1.5, markersize=4)
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
            
            self.d_psi = np.gradient(self.Psi, self.mesh.x_no.reshape((self.mesh.Nn,)))
            self.d_2_psi = np.gradient(self.d_psi, self.mesh.x_no.reshape((self.mesh.Nn,)))

            res = self.d_2_psi + (-1*self.n + self.p + self.params.dop ) 
            d_res = np.linalg.norm(res)
     
            delta = psi - self.Psi

            d_psi = np.linalg.norm(delta)
            iteration += 1

            print('ITERATION: ' + str(iteration) + '\tDELTA: ' + str(d_psi) + '\tRESIDUAL: ' + str(d_res))

    def run_time_step(self):
        #ax1 = plt.subplot(111)
        n = self.n
        p = self.p

        self.build_matrices('non_eq', 'p')
        self.solve('non_eq', 'p')

        self.build_matrices('non_eq', 'n')
        self.solve('non_eq', 'n') 

        self.dn_dt = (n - self.n) / self.dt
        self.dp_dt = (p - self.p) / self.dt

        #plt.pause(0.0001)
        #ax1.clear()

if __name__ == '__main__':
    pdd = pdd(100,0.0)

    for i in range(10):
        pdd.run_equilibrium(1000,1e-3)
        pdd.run_time_step()
        pdd.V +=1

    plt.cla()
    plt.ioff()

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax2.plot(pdd.mesh.x_no * pdd.mesh.Ldi, pdd.Psi *kT_q)
    ax1.plot(pdd.mesh.x_no * pdd.mesh.Ldi, pdd.p * n_i, label='hole concentration')
    ax1.plot(pdd.mesh.x_no * pdd.mesh.Ldi, pdd.n * n_i, label='electron concentration')

    ax2.set_title('1D Potential in a n-p Junction')
    ax2.set_xlabel('Distance (cm)')
    ax2.set_ylabel('Potential (V)')
    ax1.set_title('1D Hole and Electron Concentrations')
    ax1.set_xlabel('Distance (cm)')
    ax1.set_ylabel('Concentration (cm^-3)')
    ax1.legend()

    plt.show()