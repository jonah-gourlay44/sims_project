function [ae]=dNi_dNj_int_cont_line_Ver_1(xt)
% Integral contribution of a scalar (nodal) line (1-D)
% integral: int(dNi/dx * dNj/dx * dL)
% analytical method (Jasmin, FEM_Integral_Contributions_1D.docx)
% Input: 
% xt(2,1) - nodal coordinates of the line element
% Output
% ae(2,2) - integral contributions of the line element

% Element Length
Le=xt(2)-xt(1);

if Le<0 
   fprintf('Negative Length of the Line Element\n');
end

% Integral contributions
ae(1,1:2)=1/Le*[1,-1];
ae(2,1:2)=1/Le*[-1,1];

return;
