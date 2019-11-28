import argparse
from geometry_mesh_study import *
import compute_qfp
from model_parameters import model_parameters

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
                        default=3
                        help='upper bound for terminating potential correction norm')
    args = parser.parse_args()

    #create list of numbers of elements for mesh
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
        psi = np.random.rand(1,N1)
        phi_v = compute_hqfp(psi, mesh)
        phi_n = compute_eqfp(psi, mesh)
        parameters=model_parameters(geometry_mesh, (psi, phi_v, phi_p))
        norm_d_phi = 10**3

        #for storing field data
        We1_=np.zeros((n1_,1))
        We2_=np.zeros((n1_,1))
        We3_=np.zeros((n1_,1))
        
        iteration = 0

        #iterate through:
        while(norm_d_phi > cutoff_norm and iteration < 100)
            #perform FEM analysis to solve for d_phi
            fem = fem_study(args, parameters, geometry_mesh)
            psi = fem.d_phi + psi
            phi_v = compute_hqfp(psi, mesh)
            phi_n = compute_eqfp(psi, mesh)
            parameters.update_potentials((psi, phi_v, phi_n))
            norm_d_phi = np.norm(fem.d_phi)
            iteration += 1

        # compute new fields
        fields = field_computation(geometry_mesh, parameters, phi)
        fields.compute_fields()
        We1_[ind_] = fields.We1
        We2_[ind_] = fields.We2
        We3_[ind_] = fields.We3

        #plot these bois


if __name__ == '__main__':
    main()