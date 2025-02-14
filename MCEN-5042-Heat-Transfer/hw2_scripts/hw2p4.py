import numpy as np
import matplotlib.pyplot as plt

def create_mesh(a_L, a_num_nodes):
    mesh = np.linspace(0, a_L, a_num_nodes, endpoint=True)
    dx = a_L / a_num_nodes
    return mesh, dx

def assemble_matrix(a_T1, a_T2, a_T_inf, a_k, a_h, a_D, a_num_nodes, a_dx):
    # only supports dirichlet boundary conditions
    matrix = np.zeros([a_num_nodes, a_num_nodes])
    vector = np.zeros([a_num_nodes])
    for i in range(a_num_nodes):
        if (i == 0) or (i == a_num_nodes-1):
            # fills A matrix for Dirichlet BC
            matrix[i,i] = 1
            # fills b vector for Dirichlet BC
            if i == 0: vector[i] = a_T1
            else: vector[i] = a_T2
        else:
            # fills A matrix using centered finited difference methods
            matrix[i,i-1] = 1
            matrix[i,i+1] = 1
            matrix[i,i] = -1*(2 + (4 * a_h * a_dx**2) / (a_k * a_D))
            # fills b vector 
            vector[i] = -4 * a_h * a_dx**2 * a_T_inf / (a_k * a_D)
    return matrix, vector

def main():
    # Constants
    T_L = 40
    T_R = 180
    k = 350
    h = 40
    T_inf = 300
    D = 0.001
    L = 0.1
    N = 5
    M = 100

    mesh_N, dx_N = create_mesh(L,N)
    mesh_M, dx_M = create_mesh(L, M)

    A_N, b_N = assemble_matrix(T_L, T_R, T_inf, k, h, D, N, dx_N)
    A_M, b_M = assemble_matrix(T_L, T_R, T_inf, k, h, D, M, dx_M)

    T_N = np.linalg.solve(A_N, b_N)
    T_M = np.linalg.solve(A_M, b_M)

    # plotting part a
    labels = []
    plt.plot(mesh_N, T_N)
    labels.append('nodes = ' + str(N))
    plt.plot(mesh_M, T_M)
    labels.append('nodes = ' + str(M))
    plt.legend(labels)
    plt.xlabel('x-position [m]',fontweight='bold',fontsize=14)
    plt.ylabel('temperature [deg C]',fontweight='bold',fontsize=14)
    plt.title('Numerical Temperature Analysis: Part a',fontweight='bold',fontsize=16)
    plt.savefig('hw2p4a.png')
    plt.close()

    # part b
    h = 0
    A_M, b_M = assemble_matrix(T_L, T_R, T_inf, k, h, D, M, dx_M)
    T_M = np.linalg.solve(A_M, b_M)
    labels = []
    labels.append('nodes = ' + str(M))
    plt.plot(mesh_M, T_M)
    plt.legend(labels)
    plt.xlabel('x-position [m]', fontweight='bold', fontsize=14)
    plt.ylabel('temperature [deg C]', fontweight='bold', fontsize=14)
    plt.title('Numerical Temperature Analysis: Part b', fontweight='bold', fontsize=16)
    plt.savefig('hw2p4b.png')
    plt.close()

    # part c
    h = 40
    T_inf_c1 = 0
    T_inf_c2 = 300
    A_c1, b_c1 = assemble_matrix(T_L, T_R, T_inf_c1, k, h, D, M, dx_M)
    A_c2, b_c2 = assemble_matrix(T_L, T_R, T_inf_c2, k, h, D, M, dx_M)
    T_c1 = np.linalg.solve(A_c1, b_c1)
    T_c2 = np.linalg.solve(A_c2, b_c2)
    labels = []
    labels.append('T_inf_c1 = ' + str(T_inf_c1))
    plt.plot(mesh_M, T_c1)
    labels.append('T_inf_c1 = ' + str(T_inf_c2))
    plt.plot(mesh_M, T_c2)
    plt.legend(labels)
    plt.xlabel('x-position [m]', fontweight='bold', fontsize=14)
    plt.ylabel('temperature [deg C]', fontweight='bold', fontsize=14)
    plt.title('Numerical Temperature Analysis: Part c', fontweight='bold', fontsize=16)
    plt.savefig('hw2p4c.png')
    plt.close()


if __name__ == '__main__':
    main()
