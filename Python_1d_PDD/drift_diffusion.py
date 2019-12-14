import numpy as np 
from constants import *

#computes U(x) across the domain given p and n
def hsr_recombination_current(p,n):
    u = []
    for i in range(len(p)):
        u.append(1/(tau_n * (p[i] + n_i * np.cosh(E_T/(kT))) + tau_p*(n[i] + n_i*np.cosh(E_T/(kT)))))
        
    return u

#computes slotboom variables from qfp's across the whole domain 
def qfp_to_slotboom(eqfp, hqfp):
    u_n = []
    u_p = []
    for i in range(len(eqfp)):
        u_n.append(n_i * np.exp(-eqfp[i] / kT_q))
        u_p.append(n_i * np.exp(hqfp / kT_q))
    return (u_n, u_p)



if __name__ == '__main__':
    print(hsr_recombination_current([12,14,15,16],[12,14,15,16]))