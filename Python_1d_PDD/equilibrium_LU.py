import numpy as np
import matplotlib.pyplot as plt

q = 1.6e-19
kb = 1.38e-23
eps = 1.05e-12
T=300
ni = 1.45e10
Vt = kb*T/q
RNc = 2.82e-19
dEc = Vt*np.log(RNc/ni)

Na=1e16
Nd=1e16

Vbi = Vt*np.log(Na*Nd/(ni**2))
W=np.sqrt(2*eps*(Na+Nd)*Vbi/(q*Na*Nd))
Wn=W*np.sqrt(Na/(Na+Nd))
Wp=W*np.sqrt(Nd/(Na+Nd))
Wone=np.sqrt(2*eps*Vbi/(q*Na))
E_p=q*Nd*Wn/eps
Ldn=np.sqrt(eps*Vt/(q*Nd))
Ldp=np.sqrt(eps*Vt/(q*Na))
Ldi=np.sqrt(eps*Vt/(q*ni))

x_max=0
if(x_max < Wn):
    x_max=Wn
if(x_max < Wp):
    x_max=Wp
x_max = 20*x_max

dx=Ldn
if(dx > Ldp):
    dx=Ldp
dx=dx/20

n_max=x_max/dx
n_max=int(round(n_max))

dx=dx/Ldi

dop = np.zeros((n_max,))
for i in range(n_max):
    if i <= n_max / 2:
        dop[i]=-Na/ni
    elif i > n_max/2:
        dop[i]=Nd/ni

fi = np.zeros((n_max,))
n = np.zeros((n_max,))
p = np.zeros((n_max,))
for i in range(n_max):
    zz = 0.5*dop[i]
    if zz > 0:
        xx = zz*(1+np.sqrt(1+1/(zz**2)))
    elif zz < 0:
        xx = zz*(1-np.sqrt(1+1/(zz**2)))
    fi[i] = np.log(xx)
    n[i] = xx
    p[i] = 1/xx

delta_acc = 1e-5

dx2 = dx**2
a = np.zeros((n_max,))
b = np.zeros((n_max,))
c = np.zeros((n_max,))
f = np.zeros((n_max,))
for i in range(n_max):
    a[i] = 1/dx2
    c[i] = 1/dx2
    b[i] = -(2/dx2+np.exp(fi[i])+np.exp(-fi[i]))
    f[i] = np.exp(fi[i]) - np.exp(-fi[i]) - dop[i] - fi[i]*(np.exp(fi[i]) + np.exp(-fi[i]))

a[0] = 0
c[0] = 0
b[0] = 1
f[0] = fi[0]
a[-1] = 0
c[-1] = 0
b[-1] = 1
f[-1] = fi[-1]

flag_conv = 0
k_iter = 0
alpha = np.zeros((n_max,))
beta = np.zeros((n_max,))
v = np.zeros((n_max,))
delta = np.zeros((n_max,))
while(not flag_conv):
    k_iter += 1

    alpha[0] = b[0]
    for i in range(1,n_max):
        beta[i] = a[i]/alpha[i-1]
        alpha[i] = b[i] - beta[i]*c[i-1]
    
    v[0] = f[0]
    for i in range(1,n_max):
        v[i] = f[i] - beta[i]*v[i-1]

    temp = v[-1]/alpha[-1]
    delta[-1] = temp - fi[-1]
    fi[-1] = temp
    for i in range(n_max-2,-2,-1):
        temp = (v[i] - c[i]*fi[i+1])/alpha[i]
        delta[i] = temp - fi[i]
        fi[i] = temp
    
    delta_max = 0

    for i in range(n_max):
        xx = np.abs(delta[i])
        if xx > delta_max:
            delta_max = xx
    
    if delta_max < delta_acc:
        flag_conv = 1
    else:
        for i in range(1,n_max-1):
            b[i] = -(2/dx2+np.exp(fi[i]) + np.exp(-fi[i]))
            f[i] = np.exp(fi[i]) - np.exp(-fi[i]) - dop[i] - fi[i]*(np.exp(fi[i]) + np.exp(-fi[i]))

xxl = np.zeros((n_max,))
xxl[0] = dx*1e4

for i in range(1,n_max-1):
    xxl[i] = xxl[i-1] + dx*Ldi*1e4

xxl[-1] = xxl[-2] + dx*Ldi*1e4

plt.plot(xxl,fi)
plt.show()

