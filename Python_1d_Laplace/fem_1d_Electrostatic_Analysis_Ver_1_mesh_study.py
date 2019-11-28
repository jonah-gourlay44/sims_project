import numpy as np
from fem_functions import find_boundaries_1d
import argparse
import matplotlib.pyplot as plt
from geometry_mesh_study import *
from model_parameters import *
from matrix_assembly_es_1d import *
from scipy import sparse

Px1=100;Px2=800;Py1=100;Py2=320

def main():
    parser = argparse.ArgumentParser(description='fem 1d Electrostatic Analysis')

    parser.add_argument('--plot_mesh',
                        type=bool,
                        required=False,
                        default=False,
                        help='plot the mesh?')
    parser.add_argument('--plot_mat_props',
                        type=bool,
                        required=False,
                        default=False,
                        help='plot matrix?')
    parser.add_argument('--plot_source_field',
                        type=bool,
                        required=False,
                        default=False,
                        help='plot the source field?')
    parser.add_argument('--plot_final_solution',
                        type=bool,
                        required=False,
                        default=False,
                        help='plot the final solution?')

    args = parser.parse_args()

    linear_elements=1
    qudratic_elements=0

    num_elem=[]
    for i in range(2,201,2):
        num_elem.append(i)
    num_elem=np.array(num_elem).reshape(len(num_elem),1)

    (n1_,m)=num_elem.shape

    We1_=np.zeros((n1_,1))
    We2_=np.zeros((n1_,1))
    We3_=np.zeros((n1_,1))

    for ind_ in range(n1_):
        N1=num_elem[ind_][0]

        geometry_mesh = geometry_mesh_study(N1)
        geometry_mesh.discretize()
    
        x_min_bc,x_max_bc=find_boundaries_1d(geometry_mesh.x_no)

        #plot mesh

        parameters = model_parameters(geometry_mesh)

        #plot matrix

        matrix_assembly = matrix_assembly_1d(geometry_mesh, parameters)

        matrix_assembly.build_matrices()
        matrix_assembly.impose_boundary_conditions()

        matrix_dict = matrix_assembly.A
        b = matrix_assembly.b

        keys, values = matrix_dict.items()
        i, j = zip(*keys)
        i = np.array(i)
        j = np.array(j)
        s = np.array(list(values))

        Nn = geometry_mesh.Nn
        A = sparse.bsr_matrix((s, (i, j)), shape=(Nn, Nn), dtype=np.float)

        X = sparse.linalg.spsolve(A, b)

        Phi = X

if __name__ == '__main__':
    main()