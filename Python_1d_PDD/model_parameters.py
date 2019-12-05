import numpy as np

#some real hardcore unit analysis needs to occur at some point
class model_parameters(object):

    def __init__(self, geometry, potentials):
        self.eps_0=8.85e-12
        self.geometry = geometry
        #set initial potential functions
        self.psi, self.phi_v, self.phi_n = potentials
        

        #electron and hole mobilities
        #are these dependent on the doping?? probably??
        self.mu_n = -1
        self.mu_p = 1
        
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
        self.psi_1=1.5
        self.psi_2=0

        self.mu_r=np.zeros((geometry.Ne_1d,1))
        self.sig=np.zeros((geometry.Ne_1d,1))
        self.eps_r=np.zeros((geometry.Ne_1d,1))
        self.rho=np.zeros((geometry.Ne_1d,1))
        self.n_acceptor=np.zeros((geometry.Ne_1d,1))
        self.n_donor=np.zeros((geometry.Ne_1d,1))
        self.psi_pp=np.zeros((geometry.Ne_1d,1))

        for i in range(geometry.Ne_1d):
            if geometry.x_ec[i] < geometry.L_n:
                self.mu_r[i]=mu_r_n
                self.sig[i]=sig_n
                self.eps_r[i]=eps_r_n
                self.rho[i]=rho_n
                self.n_acceptor[i]=0
                self.n_donor[i]=n_donor_ntype
            if geometry.x_ec[i] > geometry.L_n:
                self.mu_r[i]=mu_r_p
                self.sig[i]=sig_p
                self.eps_r[i]=eps_r_p
                self.rho[i]=rho_p
                self.n_acceptor[i]=n_acceptor_ptype
                self.n_donor[i]=0

    def update_potentials(self, potentials):
        self.psi, self.phi_v, self.phi_n = potentials
        self.psi[0] = self.psi_1
        self.psi[1] = self.psi_2
        
        for i in range(self.geometry.Ne_1d):
         #approximate psi'' as constant across element
            nds_= np.asarray(self.geometry.el_1d_no[i],dtype=np.int)
            xl=self.geometry.x_no[nds_]
            Le = xl[1]-xl[0]
            Le = self.geometry.x_no[nds_[1]] - self.geometry.x_no[nds_[0]]
            if i == self.geometry.Ne_1d - 1: #special approximation of psi'', subtract slope from previous slope
                self.psi_pp[i] = (self.psi[nds_[1]] - 2*self.psi[nds_[0]] + self.psi[nds_[0]-1])/(2*Le**2)
            elif i == 0: #special approximation between next slope and this slope
                self.psi_pp[i] = (self.psi[nds_[1]+1] - 2*self.psi[nds_[1]] + self.psi[nds_[0]])/(2*Le**2)            
            else: #approximate psi'' by subtracting next slope from previous slope
                self.psi_pp[i] = (self.psi[nds_[1]+1] - self.psi[nds_[1]] - self.psi[nds_[0]] + self.psi[nds_[0]-1])/(2* Le**2)