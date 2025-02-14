import numpy as np
import matplotlib.pyplot as plt

def create_temperature_array(a_nodes_x, a_nodes_y, a_dirichlet_BC):
    T_top = a_dirichlet_BC[0]
    T_left = a_dirichlet_BC[1]
    T_right = a_dirichlet_BC[2]
    T_bottom = a_dirichlet_BC[3]
    T = np.zeros([a_nodes_x, a_nodes_y])
    T[(a_nodes_y - 1):, :] = T_top
    T[0, :] = T_bottom
    T[:, (a_nodes_x - 1):] = T_right
    T[:, 0] = T_left
    return T

def main():
    dirichlet_boundary = [100, 60, 60, 20]
    max_iterations = 500
    # Define resolution in x and y dimensions
    nodes_x = 50
    nodes_y = 50
    # Define lengths of rectangular box
    L_x = 1
    L_y = 1
    x_position = np.linspace(0, L_x, nodes_x, endpoint=True)
    y_position = np.linspace(0, L_y, nodes_y, endpoint=True)

    T = create_temperature_array(nodes_x, nodes_y, dirichlet_boundary)
    for iteration in range(max_iterations):
        for i in range(1, nodes_x-1):
            for j in range(1, nodes_y-1):
                T[i,j] = (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1]) / 4

        plt.contourf(x_position, y_position, T, 50, cmap='coolwarm')
        plt.pause(0.05)
    
    plt.colorbar()
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()