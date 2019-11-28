import numpy as np

class model_parameters(object):

    def __init__(self, geometry, potentials):
        self.eps_0=8.85e-12

        #set initial potential functions
        self.psi, self.phi_v, self.phi_n = potentials

        #parameters of n-type
        mu_r_n =1
        sig_n=0
        eps_r_n=1
        rho_n=0
        n_donor_ntype = 10**16 #units are per cm^3

        #parameters of p-type
        mu_r_p=1
        sig_p=0
        eps_r_p=0
        rho_p=0
        n_acceptor_ptype = 10**16 #units are per cm^3

        #boundary conditions (Dirichlet)
        self.phi_1=1.5
        self.phi_2=0

        self.mu_r=np.zeros((geometry.Ne_1d,1))
        self.sig=np.zeros((geometry.Ne_1d,1))
        self.eps_r=np.zeros((geometry.Ne_1d,1))
        self.rho=np.zeros((geometry.Ne_1d,1))
        self.n_acceptor=np.zeros((geometry.Ne_1d,1))
        self.n_donor=np.zeros((geometry.Ne_1d,1))


        for i in range(geometry.Ne_1d):
            if geometry.x_ec[i] < geometry.L_a:
                self.mu_r[i]=mu_r_n
                self.sig[i]=sig_n
                self.eps_r[i]=eps_r_n
                self.rho[i]=rho_n
                self.n_acceptor[i]=0
                self.n_donor[i]=n_donor_ntype
            if geometry.x_ec[i] > geometry.L_a:
                self.mu_r[i]=mu_r_p
                self.sig[i]=sig_p
                self.eps_r[i]=eps_r_p
                self.rho[i]=rho_p
                self.n_acceptor[i]=n_acceptor_ptype
                self.n_donor[i]=0

    def update_potentials(self, potentials):
        self.psi, self.phi_v, self.phi_n = potentials



