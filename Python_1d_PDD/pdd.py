import numpy as np
import itertools
from geometry_mesh_study import geometry_mesh_study
from model_parameters import parameters
from electrical_analysis_rough import compute_qfp
from scipy import integrate
from scipy import sparse
from scipy.sparse.linalg import spsolve

class pdd(object):

    def __init__(self, N1, V):
        self.V = V

        self.mesh = geometry_mesh_study(N1)
        self.mesh.discretize()
        self.params = parameters(self.mesh, self.V)
        self.analysis = compute_qfp(self.mesh, self.V)

        self.Psi = np.linspace(self.params.psi_1, self.params.psi_2, self.mesh.Nn)
        self.Phi_n = self.analysis.Phi_p
        self.Phi_p = self.analysis.Phi_n

        self.K = None
        self.b = None

    def dNi_dNj_beta_Ni_Nj(self, le, psi, phi_n, phi_p, n):
        x_j = le[1]; x_i = le[0]
        Le = x_j - x_i

        x = np.linspace(x_i, x_j, n).reshape((n,))
        psi = np.linspace(psi[0], psi[1], n)
        phi_n = np.linspace(phi_n[0], phi_n[1], n)
        phi_p = np.linspace(phi_p[0], phi_p[1], n)

        beta = -(np.exp(psi - phi_n) + np.exp(phi_p - psi))
        dNi = -1/Le; dNj = 1/Le
        Ni = 1/Le * (x_j - x); Nj = 1/Le * (x - x_i)

        ae = np.zeros((2,2))
        ae[0,0] = integrate.simps(dNi ** 2 + beta * Ni ** 2, x)
        ae[0,1] = integrate.simps(dNi * dNj + beta * Ni * Nj, x)
        ae[1,0] = ae[0,1]
        ae[1,1] = integrate.simps(dNj ** 2 + beta * Nj ** 2, x)

        return ae
    
    def Ni_f(self, le, psi, phi_n, phi_p, ind, n):
        x_j = le[1]; x_i = le[0]
        Le = x_j - x_i

        x = np.linspace(x_i, x_j, n).reshape((n,))
        psi = np.linspace(psi[0], psi[1], n)
        phi_n = np.linspace(phi_n[0], phi_n[1], n)
        phi_p = np.linspace(phi_p[0], phi_p[1], n)

        d_2_psi = np.gradient(np.gradient(psi))
        N_a = self.params.N[ind][0]
        N = np.ones((n,)) * N_a
        f = -d_2_psi + np.exp(psi - phi_n) - np.exp(phi_p - psi) - N
        Ni = 1/Le * (x_j - x); Nj = 1/Le * (x - x_i)

        be = np.zeros((2,1))
        be[0,0] = integrate.simps(Ni * f, x)
        be[1,0] = integrate.simps(Nj * f, x)

        return be     

    def build_matrices(self):
        n = 100
        Ne_1d = self.mesh.Ne_1d
        el_1d_no = self.mesh.el_1d_no
        Nn = self.mesh.Nn
        x_no = self.mesh.x_no

        A = {}
        b = np.zeros((Nn,1))

        for i in range(Ne_1d):
            nds_= np.asarray(el_1d_no[i],dtype=np.int)
            xl=x_no[nds_]

            ae = self.dNi_dNj_beta_Ni_Nj(xl, self.Psi[nds_], self.Phi_n[nds_], self.Phi_p[nds_], n)
            be = self.Ni_f(xl, self.Psi[nds_], self.Phi_n[nds_], self.Phi_p[nds_], i, n)

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

            for i in range(Nn):
                index_Nn = (Nn-1, i)
                index_1 = (0, i)

            A[index_Nn] = 0; A[index_1] = 0

            A[(0,0)] = 1; b[0] = 0
            A[(Nn-1, Nn-1)] = 1; b[Nn-1] = 0
            keys, values = zip(*A.items())
            i, j = zip(*keys)
            i = np.array(i)
            j = np.array(j)
            s = np.array(values)

            self.K = sparse.csr_matrix((s, (i, j)), shape=(Nn, Nn), dtype=np.float)
            self.b = b

    def solve(self):

        delta = spsolve(self.K, self.b)

        return delta

    def run(self, num_iter, cutoff):

        iteration = 0; d_psi = 1e3

        while iteration < num_iter and d_psi > cutoff:
            print('ITERATION: ' + str(iteration) + ' DELTA: ' + str(d_psi))
            self.build_matrices()

            delta = self.solve()

            self.Psi = self.Psi + delta

            self.analysis.integrate(self.Psi)
            self.Phi_n = self.analysis.Phi_n
            self.Phi_p = self.analysis.Phi_p

            if delta[0] != 'nan':
                d_psi = np.linalg.norm(delta)
            iteration += 1

if __name__ == '__main__':
    pdd = pdd(10,0.5)
    pdd.run(10,3)

    print(pdd.Psi)



            



