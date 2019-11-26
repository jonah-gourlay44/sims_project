% Field Scalar Plot 1-D

if quadratic_elements
L1=quadratic_basis_single(0.5,1);
L2=quadratic_basis_single(0.5,2);
L3=quadratic_basis_single(0.5,3);

for e=1:Ne_1d
    
    ind1=el_1d_no(e,1);ind2=el_1d_no(e,2);
    x1=x_no(ind1);x2=x_no(ind2);
    xm=(x1+x2)/2;
    Hym=L1*Hy(ind1)+L2*Hy(ind2)+L3*Hy(Nn+e);
    plot([x1,xm,x2],[abs(Hy(ind1)),abs(Hym),abs(Hy(ind2))],'r');
 
end
end

if linear_elements

for e=1:Ne_1d
    
    ind1=el_1d_no(e,1);ind2=el_1d_no(e,2);
    x1=x_no(ind1);x2=x_no(ind2);
    plot([x1,x2],[V(ind1),V(ind2)],'Color',line_color);
 
end
end
