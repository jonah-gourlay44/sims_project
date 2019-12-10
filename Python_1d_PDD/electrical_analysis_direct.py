import numpy as np
import itertools
from scipy import integrate
from geometry_mesh_study import geometry_mesh_study
from model_parameters import model_parameters

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
kT = 1e-10

L_D = np.sqrt(kT_q * eps / n_i)

class compute_qfp(object):

    def __init__(self, mesh, V):
        self.Nn = mesh.Nn
        self.R = mesh.L_n + mesh.L_p

        self.Phi_p = np.random.rand(self.Nn)
        self.Phi_n = np.random.rand(self.Nn)
        self.Phi_p[0] = V / kT_q; self.Phi_p[self.Nn-1] = V / kT_q
        self.Phi_n[0] = V / kT_q; self.Phi_n[self.Nn-1] = V / kT_q

        self.Psi = None
        self.E_f = None
        self.constants = {}

        self.J_p_const = None
        self.Phi_t = np.exp(E_T / kT)

        self.x_no = mesh.x_no
        self.el_1d_no = mesh.el_1d_no

    def interpolate(self, func, x, xt):
        [ind_1, ind_2] = xt
        [func_i, func_j] = [func[ind_1], func[ind_2]]
        [x_1, x_2] = self.x_no[xt]
        return func_i + (x - x_1) * (func_j - func_i)/(x_2 - x_1)
    
    def get_nodes(self, x):
        x_no = self.x_no
        el_1d_no = np.asarray(self.el_1d_no, dtype=np.int)
        ind_ = el_1d_no[np.where((x_no[el_1d_no][:,0] <= x) & (x_no[el_1d_no][:,1] >= x))]
        if len(el_1d_no) == 1:
            return el_1d_no[ind_][0]
        elif x >= self.R:
            return el_1d_no[len(el_1d_no)-1]
        return el_1d_no[0]

    def E(self, x):
        [x_1, x_2] = self.get_nodes(x)
        return self.interpolate(self.E_f, x, [x_1, x_2])

    def phi_p(self, x):
        [x_1, x_2] = self.get_nodes(x)
        return self.interpolate(self.Phi_p, x, [x_1, x_2])

    def phi_n(self, x):
        [x_1, x_2] = self.get_nodes(x)
        return self.interpolate(self.Phi_n, x, [x_1, x_2])
    
    def psi(self, x):
        [x_1, x_2] = self.get_nodes(x)
        return self.interpolate(self.Psi, x, [x_1, x_2])
    
    def u(self, x):
        tau_p_term = tau_p * (np.exp(self.psi(x) - self.phi_n(x)) + np.exp(self.Phi_t))
        tau_n_term = tau_n * (np.exp(self.phi_p(x) - self.psi(x)) + np.exp(self.Phi_t))
        denominator = tau_n_term + tau_p_term
        numerator = n_i * (np.exp(self.phi_p(x) - self.phi_n(x)) - 1)
        return numerator / denominator

    def U(self, x):
        D_0 = self.constants['D_0']
        return L_D ** 2 * self.u(x) / (D_0 * n_i)

    def J_r(self, x):
        return integrate.quad(lambda t: self.U(t), 0, x)[0]
    
    def gamma_p(self, x):
        A = self.constants['A']
        mu_0 = self.constants['mu_0']
        v_scat = self.constants['v_scat']
        return np.sqrt(1.0 + N / (N_c + N / A ** 2)) + self.E(x) * mu_0 / v_scat

    def FR(self, x):
        #print(integrate.quad(lambda t: self.J_r(t), x, self.R))
        return integrate.quad(lambda t: self.J_r(t) * self.gamma_p(t) * np.exp(self.psi(t)), x, self.R)[0]

    def F(self, x):
        return integrate.quad(lambda t: self.gamma_p(t) * np.exp(self.psi(t)), x, self.R)[0]

    def integrate(self, psi, case):
        self.Psi = psi
        self.E_f = -1 * np.gradient(psi)

        switcher = {
            'p': [mu_p, A_p, mu_0_p, v_scat_p, D_0_p],
            'n': [mu_n, A_n, mu_0_n, v_scat_n, D_0_n]
        }

        [mu, A, mu_0, v_scat, D_0] = switcher[case]

        self.constants = {
            'mu': mu,
            'A': A,
            'mu_0': mu_0,
            'v_scat': v_scat,
            'D_0': D_0
        }

        if case == 'p':
            self.J_p_const = (np.exp(self.Phi_p[0]) - np.exp(self.Phi_p[self.Nn-1]) + self.FR(0)) / self.F(0)
            for i,x in enumerate(self.x_no):
                if i != 0 and i != Nn-1:
                    self.Phi_p[i] = np.log(self.J_p_const * self.F(x) - self.FR(x) + np.exp(self.Phi_p[self.Nn-1]))

        #elf.J_e_const = (-no.exp(-self.Phi_n[0]) + 1 + self.FR_e(0)) / self.F_e(0)
    
if __name__ == '__main__':
    mesh = geometry_mesh_study(20)   
    mesh.discretize()

    Nn = mesh.Nn

    phi_n = np.random.rand(Nn)
    phi_p = np.random.rand(Nn)
    psi = np.random.rand(Nn)
    potentials = [phi_p, phi_n, psi]

    params = model_parameters(mesh, potentials)

    analysis = compute_qfp(mesh, 2)

    x_no = mesh.x_no
    el_1d_no = np.asarray(mesh.el_1d_no, dtype=np.int)

   
    analysis.integrate(psi, 'p')

    print(analysis.Phi_p)


    