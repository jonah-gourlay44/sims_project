import numpy as np

#Environment Constants
kT_q = 0.025875 # T = 300; q = 1.6e-19; k = 1.38e-23; kT_q := k*T/q
kT = 1e-10; q_Na = 0.0016; q_Nd = 0.0016; q_ni = 2.4e-9; q = 1e-19

#Material Constants
E_T = 0; tau_p = 2e-6; tau_n = 1e-5; mu_p = 1400; mu_n = 450; eps = 8.84e-14 * 11.6; n_i = 8.954e9
A_n = 1305; A_p = 402; mu_0_n = 1400; mu_0_p = 450; v_scat_p = 7.5e6; v_scat_n = 1e7
N_c = 2.8e19; D_0_p = 13; D_0_n = 28; N_a = 1e16; N_d = 1e16; N = 1e16; mu_p = 450; mu_n = 1400; 
D_n = 36.192; D_p = 11.633; W = 4.22e-5; Wn = 2.11e-5; Wp = 2.11e-5; D_0 = 11.896; mu_0 = 460

V_bi = kT_q * np.log(N_a * N_d / n_i**2)



L_D = 4.339e-3

