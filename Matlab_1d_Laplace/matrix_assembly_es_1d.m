% Electrostatic Solver 1-D
% Matrix Assembly
% August 12, 2019

A=sparse(Nn,Nn);        % FEM matrix
b=zeros(Nn,1);          % RHS
X=zeros(Nn,1);          % Unknown vector

for i=1:Ne_1d % Loop over 1-D elements

    nds_=el_1d_no(i,1:2);
    xl=x_no(nds_);
    
    ae=dNi_dNj_int_cont_line_Ver_1(xl); % ae(2,2)
    be=Ni_int_cont_line_Ver_1(xl); % be(2,1)
    
    ae=ae*eps_r(i);
    be=be*rho(i);
    
    A(nds_,nds_)=A(nds_,nds_)+ae;
    b(nds_,1)=b(nds_,1)+be;
    
end % End of Loop over 1-D elements

A=A*eps_0;
