function [dN]=dNi_line_Ver_1(xt)
% Derivation of a scalar (nodal) shape function of a line element (1-D)
% Derivative: dN/dx
% analytical method (Jasmin, FEM_Integral_Contributions_1D.docx)
% Input: 
% xt(2,1) - nodal coordinates of the line element
% Output
% dN(2,1) - derivatives of the shape functions

% Element Length
Le=xt(2)-xt(1);

if Le<0 
   fprintf('Negative Length of the Line Element\n');
end

% Derivatives of the shape functions
dN(1,1)=-1/Le;
dN(2,1)=1/Le;

return;
