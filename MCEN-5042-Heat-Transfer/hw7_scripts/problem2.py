import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def part_d():
    N = 50001  # Number of cells
    eta_domain = [0, 50]
    eta = np.linspace(eta_domain[0], eta_domain[1], N)  # eta vector

    # non-dim stream functions
    f_0 = 0
    dfdEta_0 = 0
    ddfdEta2_0 = 0.332033  # guess (given)

    # non-dim thermal functions
    alpha = input('Enter Alpha: ')
    '''
        Pr = 0.100, alpha = 0.14 
        Pr = 1.000, alpha = 0.33
        Pr = 10.00, alpha = 0.73
        Pr = 100.0, alpha = 1.57
    '''
    g_0 = 0
    dgdEta = alpha  # guess
    Pr = 100.0

    initial_conditions = [f_0, dfdEta_0, ddfdEta2_0, g_0, dgdEta]  # initial conditions

    def func(V, eta, Pr):

        f = V[0]
        f_dot = V[1]
        f_ddot = V[2]
        f_etadot = -0.5 * f * f_ddot

        g = V[3]
        g_dot = V[4]
        dgdEta = -0.5 * Pr * f * g_dot

        dVdEta = np.zeros(5)
        dVdEta[0] = f_dot
        dVdEta[1] = f_ddot
        dVdEta[2] = f_etadot

        dVdEta[3] = g_dot
        dVdEta[4] = dgdEta

        return dVdEta


    V = odeint(func, initial_conditions, eta, args=(Pr,))
    y = V[:, 0]
    y_dot = V[:, 1]
    y_ddot = V[:, 2]
    T = V[:, 3]
    T_dot = V[:, 4]

    plt.figure(1)
    plt.clf()

    plt.subplot(3, 2, 1)
    plt.plot(y, eta, 'r-')
    plt.axvline(0, color='k')
    plt.xlabel('f')
    plt.ylabel('eta')
    plt.subplot(3, 2, 2)
    plt.plot(y_dot, eta, 'r-')
    plt.axvline(0, color='k')
    plt.axvline(1, color='k', linestyle='--')
    plt.xlabel('df/deta')
    plt.ylabel('eta')
    plt.subplot(3, 2, 3)
    plt.plot(y_ddot, eta, 'r-')
    plt.axvline(0, color='k')
    plt.axvline(1, color='k', linestyle='--')
    plt.xlabel('d2f/deta')
    plt.ylabel('eta')

    plt.subplot(3, 2, 4)
    plt.plot(T, eta, 'r-')
    plt.axvline(0, color='k')
    plt.xlabel('T*')
    plt.ylabel('eta')
    plt.subplot(3, 2, 5)
    plt.plot(T_dot, eta, 'r-')
    plt.axvline(0, color='k')
    plt.axvline(1, color='k', linestyle='--')
    plt.xlabel('dT*/deta')
    plt.ylabel('eta')
    plt.show()

def part_e():
    # delta * Pr^1/3 = delta_T >> f_2 * Pr^1/3 = f_4
    Pr_array = [0.1, 1.0, 10.0, 100.0]
    f4_array_trials = [0.14, 0.33, 0.73, 1.57]
    f4_array_estimate = []
    error_array = []
    f2 = 0.332033
    for i in range(len(Pr_array)):
        f4_array_estimate.append(f2 * Pr_array[i]**(1/3))
        error_array.append(np.abs(100 * (f4_array_estimate[i] - f4_array_trials[i]) / f4_array_trials[i]))
    print(error_array)

def main():
    # part_d()
    part_e()

if __name__ == '__main__':
    main()

