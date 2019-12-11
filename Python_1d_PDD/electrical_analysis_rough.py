import numpy as np
import itertools
from scipy import integrate
from geometry_mesh_study import geometry_mesh_study
from model_parameters import model_parameters
from fem_ep_study import fem_study

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
E_T = 0; tau_p = 2e-6; tau_n = 1e-5; mu_p = 1400; mu_n = 450; eps = 11.6; n_i = 1.5e10
A_n = 1305; A_p = 402; mu_0_n = 1400; mu_0_p = 450; v_scat_p = 7.5e6; v_scat_n = 1e7
N_c = 2.8e19; D_0_p = 13; D_0_n = 28

#Environment Constants
kT_q = 0.025875 # T = 300; q = 1.6e-19; k = 1.38e-23; kT_q := k*T/q
kT = 1e-10

L_D = np.sqrt(kT_q * eps / n_i)

class compute_qfp(object):

    def __init__(self, mesh, V):
        self.params = None
        self.Nn = mesh.Nn
        self.R = mesh.L_n + mesh.L_p
        self.V = V

        self.Phi_1 = - V / kT_q; self.Phi_Nn = 0

        #self.Phi_p = np.linspace(self.Phi_1, self.Phi_Nn, self.Nn) 
        #self.Phi_n = np.linspace(self.Phi_1, self.Phi_Nn, self.Nn) 

        self.Phi_p = np.zeros((self.Nn,))
        self.Phi_n = np.zeros((self.Nn,))

        self.Psi = None
        self.E_f = None
        self.constants = {}

        self.Phi_t = E_T / kT
        self.x_no = mesh.x_no.reshape((len(mesh.x_no),))
        self.el_1d_no = mesh.el_1d_no

    def apply_bcs(self):
        self.Phi_p[0] = self.Phi_1; self.Phi_p[-1] = self.Phi_Nn
        self.Phi_n[0] = self.Phi_1; self.Phi_n[-1] = self.Phi_Nn
    
    def integrate(self, psi, params):
        self.Psi = psi
        self.E_f = -1 * np.gradient(psi)
        self.params = params

        switcher = {
            'p': [mu_p, A_p, mu_0_p, v_scat_p, D_0_p],
            'n': [mu_n, A_n, mu_0_n, v_scat_n, D_0_n]
        }

        cases = ['p','n']

        for case in cases:

            [mu, A, mu_0, v_scat, D_0] = switcher[case]

            self.constants = {
                'mu': mu,
                'A': A,
                'mu_0': mu_0,
                'v_scat': v_scat,
                'D_0': D_0
            }

            u = n_i * (np.exp(self.Phi_p - self.Phi_n) - 1) / (tau_p * (np.exp(self.Psi - self.Phi_n) + np.exp(self.Phi_t)) + tau_n * (np.exp(self.Phi_p - self.Psi) + np.exp(-self.Phi_t)))
            U = L_D ** 2 * u / (D_0 * n_i)
            J_r = integrate.simps(U, self.x_no)

            E = -1 * np.gradient(self.Psi)
            N = self.params.N.tolist()
            N.append(self.params.N[-1])
            N = np.asarray(N) 
            gamma = np.sqrt(1 + N/((N_c / n_i) + N/A**2)) + E * mu_0 / v_scat
            
            print(self.Psi)

            integrand_FR = np.zeros(self.x_no.shape)
            integrand_F = np.zeros(self.x_no.shape)
            FR = np.zeros(self.x_no.shape)
            F = np.zeros(self.x_no.shape)

            if case == 'p':
                integrand_FR = J_r * gamma * np.exp(self.Psi)
                integrand_F = gamma * np.exp(self.Psi)
            elif case == 'n':
                integrand_FR = J_r * gamma * np.exp(-1 * self.Psi)
                integrand_F = gamma * np.exp(-1 * self.Psi)
                
            for i in range(len(self.x_no)):
                FR[i] = integrate.simps(integrand_FR[i:], self.x_no[i:])
                F[i] = integrate.simps(integrand_F[i:], self.x_no[i:])

            J_const = 0

            if case == 'p':
                J_const = (np.exp(self.V) - 1 + FR[0]) / F[0]
                self.Phi_p = np.log(J_const * F - FR + 1)
            elif case == 'n':
                J_const = (-np.exp(-1 * self.V) + 1 + FR[0]) / F[0]
                self.Phi_n = np.log(J_const * F - FR + 1)
            
            self.apply_bcs()


'''
if __name__ == '__main__':
    N1 = 20
    geometry_mesh = geometry_mesh_study(N1)
    geometry_mesh.discretize()
    #create random initial potential guess and compute initial qfp's
    analysis = compute_qfp(geometry_mesh, 0.7)
    print(analysis.Phi_p)
    psi = np.abs(np.random.rand(1,N1+1)[0])
    psi = 10*psi/np.linalg.norm(psi)#normalize our guess
    #analysis.integrate(psi)

    phi_v = analysis.Phi_p
    phi_n = analysis.Phi_n

    parameters=model_parameters(geometry_mesh, (psi, phi_v, phi_n), 0.7)

    psi[0] = parameters.psi_1
    psi[-1] = parameters.psi_2

    parameters.update_potentials((psi,phi_v,phi_n))

    for i in range(10):

        analysis.integrate(psi)

        fem = fem_study(parameters, geometry_mesh)
        print(psi)

        psi = fem.d_psi + psi
        phi_v = analysis.Phi_p
        phi_n = analysis.Phi_n
        parameters.update_potentials((psi, phi_v, phi_n))
    
        #print(phi_v)
'''
            


                    


        


