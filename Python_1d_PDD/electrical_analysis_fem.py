import numpy as np
import itertools
from scipy import integrate, sparse
from scipy.sparse.linalg import spsolve

'''
-----------------------------------------------------
Class for computing electrical quantities and fields:

Phi_p - Hole Quasi-Fermi Potential
Phi_n - Electron Quasi-Fermi Potential
J_p_const - Constant part of hole current density
J_e_const - Constant part of electron current density
y_p - Hole relative diffusion constant
y_n - Electron relative diffusion constant
J_r - Recombination current
u - Recombination rate

-----------------------------------------------------
'''

#Material Constants
E_T = 0; tau_p = 2e-6; tau_n = 1e-5; mu_p = 1400; mu_n = 450; eps = 11.6; n_i = 1e10
A_n = 1305; A_p = 402; mu_0_n = 1400; mu_0_p = 450; v_scat_p = 7.5e6; v_scat_n = 1e7
N_c = 2.8e19; N = 1e16; D_0_p = 13; D_0_n = 28

#Environment Constants
kT_q = 0.025875 # T = 300; q = 1.6e-19; k = 1.38e-23; kT_q := k*T/q

L_D = np.sqrt(kT_q * eps / n_i)

def compute_gamma(E, case):

    case_switcher = {
        'p': [A_p, mu_0_p, v_scat_p],
        'n': [A_n, mu_0_n, v_scat_n]
    }

    [A, mu_0, v_scat] = case_switcher[case]

    gamma = np.sqrt(1.0 + N/(N_c + N/A**2)) + E*mu_0/v_scat

    return gamma

def interpolate(x, ind_, func, xt):

    [func_i, func_j] = [func[ind_[0]], func[ind_[1]]]

    return func_i + (x - xt[0]) * (func_j - func_i)/(xt[1] - xt[0])

def compute_U(x, ind_, xl, Phi_p, Phi_n, Psi, case):
    switcher_D = {
        'p': D_0_p,
        'n': D_0_n
    }

    D_0 = switcher_D[case]

    phi_n = interpolate(x, ind_, Phi_n, xl)
    phi_p = interpolate(x, ind_, Phi_p, xl)
    psi = interpolate(x, ind_, Psi, xl)

    u = n_i * (np.exp(phi_p - phi_n) - 1) / (tau_p*(np.exp(psi - phi_n) + 1) + tau_n*(np.exp(phi_p - psi) + 1))

    U = L_D ** 2 * u / (D_0 * n_i)

    return U

def alpha_dNi_dNj_p(x, Le, gamma, psi):

    integrand = -(1/Le**2) * np.exp(-psi[0] - (psi[1] - psi[0])/Le*x) / (gamma[0] + (gamma[1] - gamma[0])/Le*x)

    return integrand

def alpha_dNi_dNj_n(x, Le, gamma, psi):

    integrand = -(1/Le**2) * np.exp(psi[0] + (psi[1] - psi[0])/Le*x) / (gamma[0] + (gamma[1] - gamma[0])/Le*x)

    return integrand

def Ni_f(x, ind_, xl, phi_p, phi_n, psi, case):

    Le = xl[1] - xl[0]

    U = compute_U(x, ind_, xl, phi_p, phi_n, psi, case)

    integrand = (x - xl[0]) / Le * U

    return integrand


class compute_qfp(object):

    def __init__(self, Nn, V):
        self.Phi_p = np.random.rand(Nn)
        self.Phi_n = np.random.rand(Nn)
        self.Phi_1 = V; self.Phi_Nn = 0
        self.Phi_p[0] = self.Phi_1; self.Phi_p[Nn-1] = self.Phi_Nn
        self.Phi_n[0] = self.Phi_1; self.Phi_n[Nn-1] = self.Phi_Nn

        self.Psi = None
        
        self.K = None
        self.b = None

    def build_matrices(self, mesh, parameters, psi, case):
        self.Psi = psi

        Ne_1d = mesh.Ne_1d
        el_1d_no = mesh.el_1d_no
        x_no = mesh.x_no
        Nn = mesh.Nn
        
        A = {}
        b = np.zeros((Nn,1))

        E = -1*np.gradient(psi)

        gamma = compute_gamma(E, case)

        for i in range(Ne_1d):    

            nds_= np.asarray(el_1d_no[i],dtype=np.int)
            xl=x_no[nds_]

            Le = xl[1] - xl[0]

            nds_ = tuple(nds_)
            indices = list(itertools.product(nds_,nds_))

            be = np.zeros((2,1))
            x_f_int=integrate.quad(lambda x: Ni_f(x, nds_, xl, self.Phi_p, self.Phi_n, psi, case), xl[0], xl[1])
            be[:,0] = x_f_int[0]*np.asarray([1,1]) 

            b[nds_,0] = b[nds_,0] + be.reshape((be.shape[0],)) 

            for index in indices:
                nds = np.asarray(index)

                gamma_ind_ = gamma[nds]
                psi_ind_ = psi[nds]

                if case == 'p':
                    ae=integrate.quad(lambda x: alpha_dNi_dNj_p(x, Le, gamma_ind_, psi_ind_), 0, Le)[0]
                elif case == 'n':
                    ae=integrate.quad(lambda x: alpha_dNi_dNj_n(x, Le, gamma_ind_, psi_ind_), 0, Le)[0]
                
                try:
                    val = A[index]
                except KeyError:
                    val = 0

                A[index] = val + ae
             
        for i in range(Nn):
            index_Nn = (Nn-1, i)
            index_1 = (0, i)

            A[index_Nn] = 0; A[index_1] = 0

        A[(0,0)] = 1; b[0] = np.exp(self.Phi_1)
        A[(Nn-1, Nn-1)] = 1; b[Nn-1] = np.exp(self.Phi_Nn)
        keys, values = zip(*A.items())
        i, j = zip(*keys)
        i = np.array(i)
        j = np.array(j)
        s = np.array(values)

        self.K = sparse.csr_matrix((s, (i, j)), shape=(Nn, Nn), dtype=np.float)
        self.b = b

    def solve_system(self, case):

        X = spsolve(self.K, self.b)
        #print('PRINTING X')
        #print(X)
        if case == 'p':
            self.Phi_p = np.log(X)
            #print(self.Phi_p)
        elif case == 'n':
            self.Phi_n = -np.log(X)
            #print(self.Phi_n)



