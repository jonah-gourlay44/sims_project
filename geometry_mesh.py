#Electrostatic 1-D, Geometry and Mesh
import numpy as np

L_a=0.01
L_i=0.01
N_a=60
N_i=60
dx_a=L_a/N_a
dx_i=L_i/N_i

Ne_1d=N_a+N_i
Nn=Ne_1d+1

el_1d_no=np.zeros((Ne_1d,2))
el_mat_1d=np.zeros((Ne_1d,1))
x_no=np.zeros((Nn,1))
x_ec=np.zeros((Ne_1d,1))

x_no[0]=0
n_count=0
e_count=-1

for i in range(N_a):
    n_count=n_count+1
    e_count=e_count+1
    x_no[n_count]=x_no[n_count-1]+dx_a
    el_1d_no[e_count,:]=np.asarray([n_count-1,n_count])
    el_mat_1d[e_count,0]=1
    x_ec[e_count,0]=np.mean([x_no[n_count-1],x_no[n_count]])

for i in range(N_i-1):
    n_count=n_count+1
    e_count=e_count+1
    x_no[n_count]=x_no[n_count+1]+dx_i
    el_1d_no[e_count,:]=np.asarray([n_count-1,n_count])
    el_mat_1d[e_count,0]=1
    x_ec[e_count,0]=np.mean([x_no[n_count-1],x_no[n_count]])

print(x_no)
