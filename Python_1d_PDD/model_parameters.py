import numpy as np

class model_parameters(object):

    def __init__(self, geometry):
        self.eps_0=8.85e-12

        #parameters of air
        mu_r_a =1
        sig_a=0
        eps_r_a=1
        rho_a=1e-7

        #parameters of insulator
        mu_r_i=1
        sig_i=0
        eps_r_i=2
        rho_i=0

        #boundary conditions (Dirichlet)
        self.phi_1=1.5
        self.phi_2=0

        self.mu_r=np.zeros((geometry.Ne_1d,1))
        self.sig=np.zeros((geometry.Ne_1d,1))
        self.eps_r=np.zeros((geometry.Ne_1d,1))
        self.rho=np.zeros((geometry.Ne_1d,1))

        for i in range(geometry.Ne_1d):
            if geometry.x_ec[i] < geometry.L_a:
                self.mu_r[i]=mu_r_a
                self.sig[i]=sig_a
                self.eps_r[i]=eps_r_a
                self.rho[i]=rho_a
            if geometry.x_ec[i] > geometry.L_a:
                self.mu_r[i]=mu_r_i
                self.sig[i]=sig_i
                self.eps_r[i]=eps_r_i
                self.rho[i]=rho_i



