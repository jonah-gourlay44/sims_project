import numpy as np
import scipy.integrate as integrate

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

#TODO check this
def Ni_f_int_cont_line_Ver_1(xt, Na,a,b,a_n,b_n,a_p,b_p, psi_pp,Le):
  
    be = np.zeros((2,1))
    for i in range(2):
        be[i,0] = ((a[i]*psi_pp + a[i]*N)*(xt[1]**2-xt[0]**2)/2 + 
                    np.exp(b_n)*a[i]*x_exp_ax_int(a_n, xt) -
                    a[i]*np.exp(b_p)*x_exp_ax_int(a_p, xt) -
                    b[i]*(psi_pp-N)*Le + 
                    b[i] * (np.exp(b_n)*exp_ax_int(a_n, xt) - np.exp(b_p)*exp_ax_int(a_p,xt))
                    )
    return be
    

def dNi_line_Ver_1(xt):
    Le = xt[1] - xt[0]

    if Le < 0: 
        print('Negative length of the line element')
    
    dN = np.zeros((2,1))
    dN[0,0] = -1/Le
    dN[1,0] = 1/Le

    return dN

def beta_Ni_Nj(xt, a,b,a_n,b_n,a_p,b_p):
    ae = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            ae[i,j] = ( np.exp(b_n)*(
                        a[i]*a[j]*x2_exp_ax_int(a_n,xt)+
                        (a[i]*b[j]+a[j]*b[i])*x_exp_ax_int(a_n,xt)+
                        b[i]*b[j]*exp_ax_int(a_n,xt)) +
                        np.exp(b_p)*(
                        a[i]*a[j]*x2_exp_ax_int(a_p,xt)+
                        (a[i]*b[j]+a[j]*b[i])*x_exp_ax_int(a_p,xt)+
                        b[i]*b[j]*exp_ax_int(a_p,xt))
                    )
    return ae

###some nice little helper functions to make some of the above look not awful!!
#these work!!
def x2_exp_ax_int(a, xt):
    (x_1,x_2)=xt
    return (np.exp(a*x_2) * (a * x_2*(a*x_2-2)+2) - np.exp(a*x_1)*(a*x_1*(a*x_1-2)+2))/a**3

def x_exp_ax_int(a, xt):
    (x_1,x_2)=xt
    return (np.exp(a*x_1)*(1-a*x_1)+np.exp(a*x_2)*(a*x_2-1))/a**2

def exp_ax_int(a, xt):
    (x_1,x_2)=xt
    return (np.exp(a*x_2) - np.exp(a*x_1))/a