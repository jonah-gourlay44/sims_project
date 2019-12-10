import numpy as np
from fem_functions import find_boundaries_1d
import argparse
import matplotlib.pyplot as plt
from geometry_mesh_study import *
from model_parameters import *
from matrix_assembly_es_1d import *
from compute_fields import *
from scipy import sparse
from scipy.sparse.linalg import spsolve

Px1=100;Px2=800;Py1=100;Py2=320


class fem_study:
    #takes electric potential and quasi fermi potentials in 
    # 'parameters' and mesh as arguments. Computes and stores d_phi.
    def __init__(self, parameters, mesh):
        geometry_mesh = mesh 
        linear_elements=1

        # We1_=np.zeros((n1_,1))
        # We2_=np.zeros((n1_,1))
        # We3_=np.zeros((n1_,1))
        
        x_min_bc,x_max_bc=find_boundaries_1d(geometry_mesh.x_no)

        #plot matrix

        matrix_assembly = matrix_assembly_1d(geometry_mesh, parameters)

        matrix_assembly.build_matrices()
        matrix_assembly.impose_boundary_conditions()

        matrix_dict = matrix_assembly.A
        b = matrix_assembly.b

        keys, values = zip(*matrix_dict.items())
        i, j = zip(*keys)
        i = np.array(i)
        j = np.array(j)
        s = np.array(values)

        Nn = geometry_mesh.Nn
        A = sparse.csr_matrix((s, (i, j)), shape=(Nn, Nn), dtype=np.float)

        X = spsolve(A, b)
        self.d_psi = X
