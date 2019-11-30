import argparse
from geometry_mesh_study import *
from compute_qfp import *
from model_parameters import model_parameters
from fem_study import *

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
    parser.add_argument('--cutoff_norm',
                        type=float,
                        required=False,
                        default=3,
                        help='upper bound for terminating potential correction norm')
    args = parser.parse_args()

    cutoff_norm = args.cutoff_norm

    #create list of numbers of elements for meshes
    num_elem=[]
    for i in range(2,201,2):
        num_elem.append(i)
    num_elem=np.array(num_elem).reshape(len(num_elem),1)
    (n1_,m)=num_elem.shape

    #perform ENTIRE study for each possible number of elements (???)
    for ind_ in range(n1_):
        N1 = num_elem[ind_][0]
        geometry_mesh = geometry_mesh_study(N1)
        geometry_mesh.discretize()

        #create random initial potential guess and compute initial qfp's
        psi = np.random.rand(1,N1+1)[0]
        phi_v = compute_hqfp(psi, geometry_mesh)
        phi_n = compute_eqfp(psi, geometry_mesh)
        parameters=model_parameters(geometry_mesh, (psi, phi_v, phi_n))
        norm_d_phi = 10**3

        #for storing field data
        We1_=np.zeros((n1_,1))
        We2_=np.zeros((n1_,1))
        We3_=np.zeros((n1_,1))
        
        iteration = 0

        #iterate through:
        while(norm_d_phi > cutoff_norm and iteration < 100):
            #perform FEM analysis to solve for d_phi
            fem = fem_study(args, parameters, geometry_mesh)
            psi = fem.d_phi + psi
            phi_v = compute_hqfp(psi, geometry_mesh)
            phi_n = compute_eqfp(psi, geometry_mesh)
            parameters.update_potentials((psi, phi_v, phi_n))
            norm_d_phi = np.linalg.norm(fem.d_phi)
            iteration += 1

        # compute new fields
        fields = field_computation(geometry_mesh, parameters, psi)
        fields.compute_fields()
        We1_[ind_] = fields.We1
        We2_[ind_] = fields.We2
        We3_[ind_] = fields.We3

        #plot these bois


if __name__ == '__main__':
    main()