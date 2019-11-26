% Electrostatic 1-D, Geometry and Mesh
% August 12, 2019

L_a=0.01;               % Length of the air block (m)
L_i=0.01;               % Length of the insulator block (m)
N_a=60;                 % Number of elements of the air block
N_i=60;                 % Number of element in the insulator block
dx_a=L_a/N_a;           % Element length of the air
dx_i=L_i/N_i;           % Element length of the insulator

Ne_1d=N_a+N_i;          % Number of 1-D elements 
Nn=Ne_1d+1;             % Number of nodes

el_1d_no=zeros(Ne_1d,2);
el_mat_1d=zeros(Ne_1d,1);
x_no=zeros(Nn,1);
x_ec=zeros(Ne_1d,1);

x_no(1)=0;
n_count=1;
e_count=0;

for i=1:N_a
    n_count=n_count+1;
    e_count=e_count+1;
    x_no(n_count)=x_no(n_count-1)+dx_a;
    el_1d_no(e_count,1:2)=[n_count-1,n_count];
    el_mat_1d(e_count,1)=1;
    x_ec(e_count,1)=mean([x_no(n_count-1),x_no(n_count)]);
end

for i=1:N_i
    n_count=n_count+1;
    e_count=e_count+1;
    x_no(n_count)=x_no(n_count-1)+dx_i;
    el_1d_no(e_count,1:2)=[n_count-1,n_count];
    el_mat_1d(e_count,1)=1;
    x_ec(e_count,1)=mean([x_no(n_count-1),x_no(n_count)]);
end
