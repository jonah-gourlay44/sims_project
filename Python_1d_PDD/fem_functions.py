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

def Ni_f_int_cont_line_Ver_1(xt, psi_pp, psi_lin, phi_n_lin, phi_v_lin, N):
    #numerically integrate over the element
    x_f_int = integrate.quad(lambda x : x*(- psi_pp + N + np.exp(psi_lin(x) - phi_n_lin(x)) - np.exp(phi_v_lin(x) - psi_lin(x))), xt[0], xt[1])
    f_int = integrate.quad(lambda x : (- psi_pp + N + np.exp(psi_lin(x) - phi_n_lin(x)) - np.exp(phi_v_lin(x) - psi_lin(x))), xt[0], xt[1])
    
    be = np.zeros((2,1))
    be[0,0]= - x_f_int[0] + xt[1]
    be[1,0] = x_f_int[0] - xt[0]
    be = be/(xt[1]-xt[0])

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