% This subroutine imposes boundary conditions
% August 12, 2019
% Source: 01_Electrostatic_Analysis_1D.docx

% Left: Dirichlet BC (x=0)
A(1,1:Nn)=0;A(1,1)=1;b(1)=phi_1;

% Right: Dirichlet BC (x=2*L)
A(Nn,1:Nn)=0;A(Nn,Nn)=1;b(Nn)=phi_2;
