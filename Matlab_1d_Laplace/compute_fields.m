% Electrostatic Solver 1-D
% Compute Fields
% August 13, 2019

Ex=zeros(Nn,1);          % Electric field
Dx=zeros(Nn,1);          % Electric flux density
supp=zeros(Nn,1);        % Nodal supports
We1=0;
We2=0;
We3=0;

for i=1:Ne_1d % Loop over 1-D elements

    nds_=el_1d_no(i,1:2);
    xl=x_no(nds_);
    Le=xl(2)-xl(1);
    pot=Phi(nds_);
    
    ex_e=dNi_line_Ver_1(xl); % ex_e(2,1)
    dx_e=ex_e*eps_r(i)*eps_0; % dx_e(2,1)
    
    Ex(nds_(1))=Ex(nds_(1))-(ex_e(1)*pot(1)+ex_e(2)*pot(2))*Le;
    supp(nds_(1))=supp(nds_(1))+Le;
    Ex(nds_(2))=Ex(nds_(2))-(ex_e(1)*pot(1)+ex_e(2)*pot(2))*Le;
    supp(nds_(2))=supp(nds_(2))+Le;
    
    Dx(nds_(1))=Dx(nds_(1))-(dx_e(1)*pot(1)+dx_e(2)*pot(2))*Le;
    Dx(nds_(2))=Dx(nds_(2))-(dx_e(1)*pot(1)+dx_e(2)*pot(2))*Le;
    
    Exe=-(ex_e(1)*pot(1)+ex_e(2)*pot(2));
    Dxe=-(dx_e(1)*pot(1)+dx_e(2)*pot(2));
    
    We1=We1+Exe*Dxe*Le;
    We2=We2+rho(i)*mean(pot)*Le;
    
end % End of Loop over 1-D elements

for i=1:Nn % Averaging over elements
    Ex(i)=Ex(i)/supp(i);
    Dx(i)=Dx(i)/supp(i);
end

We1=0.5*We1;
We2=0.5*We2;
We3=0.5*Dx(1)*Phi(1);

fprintf('We1=%14.6e J/m^2\n',We1);
fprintf('We2=%14.6e J/m^2\n',We2);
fprintf('We3=%14.6e J/m^2\n',We3);
