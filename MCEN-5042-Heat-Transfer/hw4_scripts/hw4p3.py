import numpy as np
import matplotlib.pyplot as plt


def update_temperature_matrix(T, T_b, q_2prime, k, mesh, dx, dy):
    M, N = mesh
    center_constant = 0.5 / (dx ** 2 + dy ** 2)

    for i in range(M):
        for j in range(N):
            if i == 0 and j == 0:
                T[i, j] = T_b
            elif i == 0 and j == N - 1:
                T[i, j] = T_b
            elif i == M - 1 and j == 0:
                T[i, j] = (0.5 * (T[i - 1, j] + T[i, j + 1]))
            elif i == M - 1 and j == N - 1:
                T[i, j] = (0.5 * (T[i - 1, j] + T[i, j - 1]))
            # update non-corner boundaries
            elif i == 0:
                T[i, j] = T_b
            elif i == M - 1:
                T[i, j] = q_2prime * dy / k + T[i - 1, j]
            elif j == 0:
                T[i, j] = T_b
            elif j == N - 1:
                T[i, j] = T_b
            # update centers
            else:
                T[i, j] = center_constant * (dy ** 2 * (T[i + 1, j] + T[i - 1, j]) +
                                             dx ** 2 * (T[i, j + 1] + T[i, j - 1]))
    return T


def problem_3():
    # domain
    domain_length = 0.8
    domain_thickness = 0.8
    base_temperature = 100

    # mesh size
    size = 11
    mesh_size = (size, size)

    # heat transfer properties
    conduction_coefficient = 1
    heat_flux = 300

    dx = domain_length / mesh_size[0]
    dy = domain_thickness / mesh_size[1]

    # initialize temperature array
    temperature_array = np.ones(mesh_size)

    # plotting domain
    x_position = np.linspace(0, domain_length, mesh_size[0])
    y_position = np.linspace(0, domain_thickness, mesh_size[1])

    # iteration definitions
    iteration_max = 100
    i_array = np.zeros(iteration_max)
    t_center = np.zeros(iteration_max)

    # for plotting center lines
    y_half_node = int(0.5 * (mesh_size[1] - 1))
    x_half_node = int(0.5 * (mesh_size[0] - 1))

    i = 0
    iteration = []
    temperature_top = []
    found_error = False
    for j in range(2):
        for i in range(iteration_max):
            temperature_array = update_temperature_matrix(temperature_array, base_temperature,
                                                          heat_flux, conduction_coefficient,
                                                          mesh_size, dx, dy)
            if j == 0 and i == iteration_max - 1:
                large_N_estimate = temperature_array[mesh_size[1] - 1, x_half_node]
                temperature_array = np.ones(mesh_size)

            if j == 1:
                iteration.append(i)
                temperature_top.append(temperature_array[mesh_size[1] - 1, x_half_node])
                error = 100 * np.abs(temperature_array[mesh_size[1] - 1,
                x_half_node] - large_N_estimate) / large_N_estimate
                if error < 0.25 and not found_error:
                    N = i
                    found_error = True
    print(large_N_estimate)
    plt.plot(iteration, temperature_top)
    save_array = [iteration, temperature_top]
    np.save('time_at_iter', save_array)
    plt.plot(N, temperature_top[N], 'o')
    plt.xlabel('iteration number')
    plt.ylabel('temperature top center')
    plt.title('problem 3 part f')
    plt.savefig('p3f.png')
    plt.show()
    plt.close()
    plt.contourf(x_position, y_position, temperature_array, 50, cmap='jet', origin='lower')
    plt.colorbar()
    plt.xlabel('x-position')
    plt.ylabel('y-position')
    plt.title('problem 3 part e')
    plt.savefig('p3e.png')
    plt.close()


def main():
    problem_3()


if __name__ == '__main__':
    main()
