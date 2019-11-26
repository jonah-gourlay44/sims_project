% 1-D Electrostatic Analysis 
% August 7, 2019, Pula, Varudella
% Source: 01_Electrostatic_Analysis_1D.docx

clear;
clc;

addpath('C:\HSR_Forschung\Projekte\HSR\Feldformulationen\Matlab\FEM\1_D_FEM_Functions\Matlab');

% Model Data
% ******************
Px1=100;Px2=800;Py1=100;Py2=320;    % Size of figures
plot_mesh=0;
plot_mat_props=0;
plot_source_field=0;
plot_final_solution=0;
linear_elements=1;
quadratic_elements=0;

eps_0=8.85e-12; % Permittivity
% Model Data
% *************************

num_elem=2:2:200;

[m,n1_]=size(num_elem);

We1_=zeros(n1_,1);
We2_=zeros(n1_,1);
We3_=zeros(n1_,1);

f1=waitbar(0,'Mesh study of the DG solver in progress...');

for ind_=1:n1_

	waitbar(ind_/n1_);    
	N1=num_elem(ind_);
    
% Define the 1-D model geometry and mesh
% The units of length are (cm)

geometry_mesh_study;

[x_min_bc,x_max_bc]=find_boundaries_1d(x_no);

if plot_mesh
    figure('Position',[Px1 Py1 Px2 Py2],'Color',[1 1 1]);
    set_figure_1;
    plot_mesh_1d;
    axis on;
    xlabel('Position (cm)');
end

% return;

Model_parameters;

if plot_mat_props
    figure('Position',[Px1 Py1 Px2 Py2],'Color',[1 1 1]);
    % set_figure_1;
    plot(x_ec,eps_r,'Color',[1,0,0]);
    axis on;
    xlabel('Position (cm)');
    ylabel('eps_r');
end

% return;

% Matrix Assembly
matrix_assembly_es_1d;

% return;

% Imposing Boundary Conditions
impose_boundary_conditions;

% Solution of Equation System (Direct Solver)
X=A\b;

Phi=X;
% return;

% Compute E, D, etc...
compute_fields;

if plot_final_solution
    f1=figure('Position',[Px1 Py1 Px2 Py2],'Color',[1 1 1]);
    a1=axes('Parent',f1);
    grid on;
    xlabel('x (cm)');
    ylabel('Phi (V)');
    hold on;
    V=X(:,1);
    line_color=[1,0,0];
    plot_scalar_field_1d;
    % axis([0,0.5,0,2]);    
end

if plot_final_solution
    f1=figure('Position',[Px1 Py1 Px2 Py2],'Color',[1 1 1]);
    a1=axes('Parent',f1);
    grid on;
    xlabel('t (s)');
    ylabel('Ex (V/m)');
    hold on;
    line_color=[1,0,0];
    V=Ex(:,1);
    line_color=[1,0,0];
    plot_scalar_field_1d;
end

if plot_final_solution
    f1=figure('Position',[Px1 Py1 Px2 Py2],'Color',[1 1 1]);
    a1=axes('Parent',f1);
    grid on;
    xlabel('t (s)');
    ylabel('Dx (C/m^2)');
    hold on;
    line_color=[0,0,1];
    V=Dx(:,1);
    line_color=[1,0,0];
    plot_scalar_field_1d;
end

We1_(ind_)=We1;
We2_(ind_)=We2;
We3_(ind_)=We3;

end % End Loop Mesh Study

close(f1);

% Analytic Solution
Wea=7.579120e-10;

f1=figure('Position',[Px1 Py1 Px2 Py2],'Color',[1 1 1]);
a1=axes('Parent',f1);
grid on;
xlabel('Number of elements');
ylabel('Electric energy (J/m^2)');
hold on;
line_color=[0,0,1];
plot(num_elem',We1_,'Color',line_color);
line_color=[1,0,0];
plot(num_elem',We2_+We3_,'Color',line_color);
line_color=[0,0,0];
plot([min(num_elem),max(num_elem)],[Wea,Wea],'Color',line_color);
% legend('W_e_1','W_e_2+W_e_3');
legend('W_e_1','W_e_2+W_e_3','Analytic');

