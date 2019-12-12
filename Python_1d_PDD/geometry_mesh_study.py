#Electrostatic 1-D, Geometry and Mesh
import numpy as np

L_D = 4.07e-6

class geometry_mesh_study(object):

    def __init__(self,N1):

        self.L_n=0.01 / L_D            #Length of the n-type block
        self.L_p=0.01 / L_D             #Length of the p-type block
        self.N_n=int(N1/2)             #Number of elements in the n-type block
        self.N_p=int(N1/2)             #Number of elements in the p-type block
        self.dx_a=self.L_n/self.N_n    #Element length in n-type
        self.dx_i=self.L_p/self.N_p    #Element length in the p-type
        self.Ne_1d=self.N_n+self.N_p   #Number of 1-D elements
        self.Nn=self.Ne_1d+1           #Number of nodes
        
        self.el_1d_no=np.zeros((self.Ne_1d,2))
        self.el_mat_1d=np.zeros((self.Ne_1d,1))
        self.x_no=np.zeros((self.Nn,1))
        self.x_ec=np.zeros((self.Ne_1d,1))

        self.x_no[0]=0
        self.n_count=0
        self.e_count=-1

    def discretize(self):
        for i in range(self.N_n):
            self.n_count=self.n_count+1
            self.e_count=self.e_count+1
            self.x_no[self.n_count]=self.x_no[self.n_count-1]+self.dx_a
            self.el_1d_no[self.e_count,:]=np.asarray([self.n_count-1,self.n_count])
            self.el_mat_1d[self.e_count,0]=1
            self.x_ec[self.e_count,0]=np.mean([self.x_no[self.n_count-1],self.x_no[self.n_count]])

        for i in range(self.N_p):
            self.n_count=self.n_count+1
            self.e_count=self.e_count+1
            self.x_no[self.n_count]=self.x_no[self.n_count-1]+self.dx_i
            self.el_1d_no[self.e_count,:]=np.asarray([self.n_count-1,self.n_count])
            self.el_mat_1d[self.e_count,0]=1
            self.x_ec[self.e_count,0]=np.mean([self.x_no[self.n_count-1],self.x_no[self.n_count]])




