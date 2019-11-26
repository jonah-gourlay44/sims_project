% 1-D Electrostatic Analysis 
% Source: Smajic_FEM_Electrostatic_Analysis_1D

clear;
clc;

% Model Data
% ******************
Px1=100;Px2=800;Py1=100;Py2=320;    % Size of figures
plot_mesh=1;
plot_mat_props=1;
plot_matrix=1;
plot_source_field=0;
plot_final_solution=1;
linear_elements=1;
quadratic_elements=0;

eps_0=8.85e-12; % Permittivity
% Model Data
% *************************

% Define the 1-D model geometry and mesh
% The units of length are (cm)

geometry_mesh;

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

if plot_matrix
    f1=figure('Color',[1 1 1]);
    a1=axes('Parent',f1);
    grid on;
    % xlabel('x (cm)');
    % ylabel('Phi (V)');
    % hold on;
    % V=X(:,1);
    % line_color=[1,0,0];
    % plot_scalar_field_1d;
    spy(A);
    % axis([0,0.5,0,2]);    
end

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
    xlabel('x (cm)');
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
    xlabel('x (cm)');
    ylabel('Dx (C/m^2)');
    hold on;
    line_color=[0,0,1];
    V=Dx(:,1);
    line_color=[1,0,0];
    plot_scalar_field_1d;
end
