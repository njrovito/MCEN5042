import numpy as np
import matplotlib.pyplot as plt

def get_initial_temp(position, length_1, length_2, temp_1, temp_2, size):
    initial_temperature = np.zeros(size)
    for i in range(len(position)):
        if position[i] <= length_1:
            initial_temperature[i] = temp_1
        elif position[i] > length_2:
            initial_temperature[i] = temp_2

    return initial_temperature

def assemble_matrix(size, total_length, T_old, dt, BC):
    A = np.zeros([size, size])
    dx2 = (total_length / size)**2

    for i in range(size):
        if 0 < i < size - 1:
            A[i, i] = -(2*dt/dx2 + 1)
            A[i, i - 1] = dt/dx2
            A[i, i + 1] = dt/dx2

        elif i == 0 and BC[0] == 'flux':
            A[i, i] = 1
            A[i, i + 1] = -1

        elif i == size - 1 and BC[2] == 'flux':
            A[i, i] = 1
            A[i, i - 1] = -1

    return A

def assemble_vector(size, total_length, T_old, dt, BC):
    b = np.zeros(size)
    dx2 = (total_length / size) ** 2
    for i in range(size):
        if 0 < i < size - 1:
            b[i] = -T_old[i]
        elif i == 0 and BC[0] == 'flux':
            b[i] = BC[1]
        elif i == size - 1 and BC[2] == 'flux':
            b[i] = BC[3]

    return b


def main():
    '''
    dT/dt = d^2T/dx^2
    '''
    L1 = 2
    L2 = 1
    T1 = 100
    T2 = 20

    Lc = L1 + L2

    boundary_conditions = ['flux', 0, 'flux', 0]


    mesh_size = 100
    T = np.zeros(mesh_size)
    x = np.linspace(0, Lc, mesh_size, endpoint=True)
    dt = 0.1

    T_old = get_initial_temp(x, L1, L2, T1, T2, mesh_size)
    A = assemble_matrix(mesh_size, Lc, T_old, dt, boundary_conditions)
    b = assemble_vector(mesh_size, Lc, T_old, dt, boundary_conditions)

    T = np.linalg.solve(A,b)

    t = 0
    count = 1
    t_end = 100
    plt.plot(x, T_old)
    III = 0
    while t < t_end:
        b = assemble_vector(mesh_size, Lc, T, dt, boundary_conditions)
        T = np.linalg.solve(A, b)
        if count % 5 == 0:

            print(III, count)
            III += 1
            plt.plot(x, T)

        t += dt
        count += 1
    plt.xlabel('position', fontsize=16)
    plt.ylabel('temperature', fontsize=16)
    plt.title('Numerical Solution', fontsize=16)
    plt.savefig('numerical_solution.png')

    return


if __name__ == '__main__':
    main()