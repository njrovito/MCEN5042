import numpy as np
import matplotlib.pyplot as plt

def update_temperature_matrix(T, T_b, T_inf, Bi, mesh, dx, dy):
    M, N = mesh
    corner_boundary = (1 / (1 + Bi))
    edge_boundary = (1 / (2 + Bi))
    center_constant = 0.5 / (dx ** 2 + dy ** 2)
    for i in range(M):
        for j in range(N):
            if i == 0 and j == 0:
                T[i,j] = T_b # corner_boundary * (Bi * T_inf + 0.5 * (T[i + 1, j] + T[i, j + 1]))
            elif i == 0 and j == N - 1:
                T[i, j] = corner_boundary * (Bi * T_inf + 0.5 * (T[i + 1, j] + T[i, j - 1]))
            elif i == M - 1 and j == 0:
                T[i, j] = T_b  # corner_boundary * (Bi * T_inf + 0.5 * (T[i - 1, j] + T[i, j + 1]))
            elif i == M - 1 and j == N - 1:
                T[i, j] = corner_boundary * (Bi * T_inf + 0.5 * (T[i - 1, j] + T[i, j - 1]))
            # update non-corner boundaries
            elif i == 0:
                T[i, j] = edge_boundary * (Bi * T_inf + T[i + 1, j] + 0.5 * (T[i, j + 1] + T[i, j - 1]))
            elif i == M - 1:
                T[i, j] = edge_boundary * (Bi * T_inf + T[i - 1, j] + 0.5 * (T[i, j + 1] + T[i, j - 1]))
            elif j == 0:
                T[i, j] = T_b
            elif j == N - 1:
                T[i, j] = T[i, j - 1]
            # update centers
            else:
                T[i, j] = center_constant * (dy**2 * (T[i + 1, j] + T[i - 1, j]) +
                                             dx**2 * (T[i, j + 1] + T[i, j - 1]))
    return T

def problem_1(a_convection_coefficient, a_conduction_coefficient):
    # mesh size
    mesh_size = (21, 21)

    # variable parameters for this problem
    convection_coefficient = a_convection_coefficient
    conduction_coefficient = a_conduction_coefficient

    # domain
    domain_length = 0.1
    domain_thickness = 0.01
    base_temperature = 300
    environment_temperature = 20

    dx = domain_length / mesh_size[0]
    dy = domain_thickness / mesh_size[1]
    biot_number = convection_coefficient * dx / conduction_coefficient

    temperature_array = np.ones(mesh_size)

    # plotting domain
    x_position = np.linspace(0, domain_length, mesh_size[0])
    y_position = np.linspace(0, domain_thickness, mesh_size[1])

    # iteration definitions
    iteration_max = 10000
    i_array = np.zeros(iteration_max)
    t_center = np.zeros(iteration_max)

    # for plotting center lines
    y_half_node = int(0.5 * (mesh_size[1] - 1))
    x_half_node = int(0.5 * (mesh_size[0] - 1))

    for i in range(iteration_max):
        temperature_array = update_temperature_matrix(temperature_array, base_temperature,
                                                      environment_temperature, biot_number,
                                                      mesh_size, dx, dy)
        # part d.4
        i_array[i] = i
        t_center[i] = temperature_array[y_half_node, x_half_node]

    print('Done iterating at iteration {} of {}'.format(i+1, iteration_max))
    label = ['x position', 'y position']
    plt.contourf(x_position, y_position, temperature_array, 50, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.title('problem 1 part d.1 ')
    plt.savefig('p1d1.png')
    plt.close()

    # part d.2
    plt.plot(x_position, temperature_array[y_half_node, :])
    label = ['x position', 'temperature']
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.title('problem 1 part d.2 ')
    plt.savefig('p1d2.png')
    plt.close()

    # part d.3
    plt.plot(y_position, temperature_array[:, x_half_node])
    label = ['y position', 'temperature']
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.title('problem 1 part d.3 ')
    plt.savefig('p1d3.png')
    plt.close()

    # part d.4
    plt.plot(i_array, t_center)
    label = ['iteration', 'center temperature']
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.title('problem 1 part d.4 ')
    plt.savefig('p1d4.png')
    plt.close()


def main():
    problem_1(300, 30)

if __name__ == '__main__':
    main()
