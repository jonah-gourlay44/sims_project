import numpy as np

def find_boundaries_1d(x_nodes):
    return np.min(x_nodes), np.max(x_nodes)

def dNi_dNj_int_cont_line_Ver_1(xt):
    Le = xt[1] - xt[0]

    if Le < 0:
        print('Negative length of the line element')
    
    ae = np.zeros((2,2))
    ae[0,:]=1/Le*np.asarray([1,-1])
    ae[1,:]=1/Le*np.asarray([-1,1])

    return ae

def Ni_int_cont_line_Ver_1(xt):
    Le = xt[1] - xt[0]

    if Le < 0:
        print('Negative length of the line element')
    
    be = np.zeros((2,1))
    be[:,0]=Le/2*np.asarray([1,1])

    return be