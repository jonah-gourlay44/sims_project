#Electrostatic 1-D, Geometry and Mesh
import numpy as np

class geometry_mesh_study(object):

    def __init__(self,N1):

        self.L_a=0.01                  #Length of the air block (m)
        self.L_i=0.01                  #Length of the insulator block (m)
        self.N_a=int(N1/2)             #Number of elements in the air block
        self.N_i=int(N1/2)             #Number of elements in the insulator block
        self.dx_a=self.L_a/self.N_a    #Element length in air
        self.dx_i=self.L_i/self.N_i    #Element length in the insulator

        self.Ne_1d=self.N_a+self.N_i   #Number of 1-D elements
        self.Nn=self.Ne_1d+1           #Number of nodes
        
        self.el_1d_no=np.zeros((self.Ne_1d,2))
        self.el_mat_1d=np.zeros((self.Ne_1d,1))
        self.x_no=np.zeros((self.Nn,1))
        self.x_ec=np.zeros((self.Ne_1d,1))

        self.x_no[0]=0
        self.n_count=0
        self.e_count=-1

    def discretize(self):
        for i in range(self.N_a):
            self.n_count=self.n_count+1
            self.e_count=self.e_count+1
            self.x_no[self.n_count]=self.x_no[self.n_count-1]+self.dx_a
            self.el_1d_no[self.e_count,:]=np.asarray([self.n_count-1,self.n_count])
            self.el_mat_1d[self.e_count,0]=1
            self.x_ec[self.e_count,0]=np.mean([self.x_no[self.n_count-1],self.x_no[self.n_count]])

        for i in range(self.N_i):
            self.n_count=self.n_count+1
            self.e_count=self.e_count+1
            self.x_no[self.n_count]=self.x_no[self.n_count-1]+self.dx_i
            self.el_1d_no[self.e_count,:]=np.asarray([self.n_count-1,self.n_count])
            self.el_mat_1d[self.e_count,0]=1
            self.x_ec[self.e_count,0]=np.mean([self.x_no[self.n_count-1],self.x_no[self.n_count]])




