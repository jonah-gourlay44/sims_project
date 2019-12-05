import numpy as np
from fem_functions import find_boundaries_1d
import matplotlib.pyplot as plt
from geometry_mesh_study import *
from model_parameters import *
from matrix_assembly_es_1d import *
from compute_fields import *
from scipy import sparse
from scipy.sparse.linalg import spsolve

Px1=100;Px2=800;Py1=100;Py2=320

class fem_analysis(object):

    def __init__(self):

        self.data = {}

    def main(self):

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

        self.data['n1_'] = n1_

        for ind_ in range(n1_):
            N1=num_elem[ind_][0]

            geometry_mesh = geometry_mesh_study(N1)
            geometry_mesh.discretize()
    
            x_min_bc,x_max_bc=find_boundaries_1d(geometry_mesh.x_no)

            self.data[str(ind_)] = {'N1': N1}

            #plot mesh
            self.data[str(ind_)]['mesh'] = [geometry_mesh.x_no, np.zeros((len(geometry_mesh.x_no,)))]

            parameters = model_parameters(geometry_mesh)

            #plot matrix
            self.data[str(ind_)]['mat props'] = [geometry_mesh.x_ec, parameters.eps_r]

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
            Phi = X

            fields = field_computation(geometry_mesh, parameters, Phi)
            fields.compute_fields()

            #plot fields
            el_1d_no = geometry_mesh.el_1d_no
            x_no = geometry_mesh.x_no
            Ex = fields.Ex
            Dx = fields.Dx
            x = []
            V = []
            E = []
            D = []
            
            for e in range(geometry_mesh.Ne_1d):
                ind1 = int(el_1d_no[e,0]); ind2 = int(el_1d_no[e,1])
               
                x1 = x_no[ind1]; x2 = x_no[ind2]
                x.append(x1); x.append(x2)
                V.append(Phi[ind1]); V.append(Phi[ind2])
                E.append(Ex[ind1]); E.append(Ex[ind2])
                D.append(Dx[ind1]); D.append(Dx[ind2])
            
            self.data[str(ind_)]['electric potential'] = [np.asarray(x), np.asarray(V)]
            self.data[str(ind_)]['electric field'] = [np.asarray(x), np.asarray(E)]
            self.data[str(ind_)]['flux density'] = [np.asarray(x), np.asarray(D)]

            We1_[ind_] = fields.We1
            We2_[ind_] = fields.We2
            We3_[ind_] = fields.We3
