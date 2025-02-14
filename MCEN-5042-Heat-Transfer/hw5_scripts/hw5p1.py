import numpy as np
import matplotlib.pyplot as plt

def get_Bn(a_inputs, n):
    T_1, q_flux, k, L, w = a_inputs
    if n % 2 != 0:
        numerator = -4 * L * q_flux #  2 * L * q_flux * (-1^n -1)
        denominator = n**2 * np.pi**2 * k * np.cosh(n * np.pi * w / L)
        Bn = -numerator / denominator
    else:
        Bn = 0
    return Bn

def check_convergence(truth, test, tol_percent):
    error = 100 * np.abs((truth - test) / truth)
    if error <= tol_percent:
        return True
    else:
        return False

def part_e(a_inputs, location, n_max, show_fig=True, save_fig=False):
    T_1, q_flux, k, L, w = a_inputs
    x, y = location
    T_array = np.zeros(n_max)
    n_array = np.zeros_like(T_array)
    T = T_1

    i = 0
    T_large = T
    converged = False
    n_converged = 'did not converge'
    T_converged = 'did not converge'
    for i in range(2):
        T = T_1 # resets T for second loop
        for n in range(1, len(T_array)):
            Bn = get_Bn(a_inputs, n)
            T += Bn * np.sin(n * np.pi * x / L) * np.sinh(n * np.pi * y / L)
            if i == 0:
                T_array[n] = T
                n_array[n] = n
            if i == 1:
                converged = check_convergence(T_large, T, 0.25)
                if converged:
                    T_converged = T
                    n_converged = n
                    break
        if i == 0:
            T_large = T

    # plot
    plt.plot(n_array, T_array)
    plt.plot(n_converged, T_converged, 'o')
    labels = ['Temp at iteration', 'Converged iteration']
    plt.legend(labels)
    plt.xlabel('n', fontsize=16)
    plt.ylabel('Temperature [deg C]', fontsize=16)
    plt.title('Temperature at iteration (hw5p1e)', fontsize=20)
    if show_fig: plt.show()
    if save_fig: plt.savefig('hw5p1e.png')
    plt.close()

    return n_converged

def part_f(a_inputs, domain, mesh_size, n_max, show_fig=True, save_fig=False):
    T_1, q_flux, k, L, w = a_inputs

    x_domain = np.linspace(domain[0, 0], domain[0, 1], mesh_size)
    y_domain = np.linspace(domain[1, 0], domain[1, 1], mesh_size)

    temperature_array = np.ones([mesh_size, mesh_size])

    for i in range(mesh_size):
        for j in range(mesh_size):
            T = T_1
            x = x_domain[j]
            y = y_domain[i]
            for n in range(n_max):
                Bn = get_Bn(a_inputs, n)
                T += Bn * np.sin(n * np.pi * x / L) * np.sinh(n * np.pi * y / L)
            temperature_array[i, j] = T

    plt.contourf(x_domain, y_domain, temperature_array, 50, cmap='jet', origin='lower')
    plt.colorbar()
    plt.xlabel('x-position', fontsize=16)
    plt.ylabel('y-position', fontsize=16)
    plt.title('Spatial temperature distribution (hw5p1f)', fontsize=16)
    if show_fig: plt.show()
    if save_fig: plt.savefig('hw5p1f.png')
    plt.close()

    return

def problem_1():
    boundary_temp = 100
    heat_flux = 300
    conductivity = 1
    length = 0.8
    width = 0.8

    # part e
    x = 0.4
    y = 0.8
    max_iter = 100 # 223: largest before sinh() overflow
    inputs = [boundary_temp, heat_flux, conductivity, length, width]
    location = [x, y]
    n_converged = part_e(inputs, location, max_iter, show_fig=False, save_fig=True)

    # part f
    square_mesh_size = 11
    domain = np.array([[0,length], [0, width]])
    part_f(inputs, domain, square_mesh_size, n_converged, show_fig=False, save_fig=True)

    return

if __name__ == '__main__':
    problem_1()