% Plot 1-D Mesh

for i=1:Ne_1d
   ind1=el_1d_no(i,1);ind2=el_1d_no(i,2);
   x1=x_no(ind1);x2=x_no(ind2);
   if el_mat_1d(i,1)==1
    line_plot(x1,0,x2,0,[1,0,0],1.5);
   end
   if el_mat_1d(i,1)==2
    line_plot(x1,0,x2,0,[0,0,1],1.5);
   end
end

for i=1:Nn
   plot(x_no(i),0,'ko'); 
end
