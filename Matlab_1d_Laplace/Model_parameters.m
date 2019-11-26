% Calculation Parameters

% Parameters of Air 
mu_r_a=1;
sig_a=0;
eps_r_a=1;
rho_a=1e-7;      % charge density (C/m^3)

% Parameters of Insulator
mu_r_i=1;
sig_i=0;
eps_r_i=2;
rho_i=0;        % charge density (C/m^3)

% Boundary Conditions (Dirichlet)
phi_1=1.5;
phi_2=0;

mu_r=zeros(Ne_1d,1);
sig=zeros(Ne_1d,1);
eps_r=zeros(Ne_1d,1);
rho=zeros(Ne_1d,1);

for i=1:Ne_1d
   if x_ec(i)<L_a % Air
       mu_r(i)=mu_r_a;
       sig(i)=sig_a;
       eps_r(i)=eps_r_a;
       rho(i)=rho_a;
   end
   if x_ec(i)>L_a % Insulator
       mu_r(i)=mu_r_i;
       sig(i)=sig_i;
       eps_r(i)=eps_r_i;
       rho(i)=rho_i;
   end
end
