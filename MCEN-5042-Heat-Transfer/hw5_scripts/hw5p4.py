import numpy as np
import matplotlib.pyplot as plt

def parameters():
    # params = [L1, L2, T1, T2, alpha]
    params   = [2, 1, 100, 20, 1]
    return params

def get_summation(theta_1, theta_2, L, x, t, n_max):
    summation = 0
    constant = 2 * (theta_1 - theta_2)
    for n in range(1, n_max):
        lam = n * np.pi
        An = constant * np.sin(lam * L) / lam
        summation += An * np.cos(lam*x) * np.exp(-lam**2 * t)

    return summation

def problem_4(show_fig=True, save_fig=False):
    # define problem
    L1, L2, T1, T2, alpha = parameters()

    mesh = 100
    X = np.linspace(0, 1, mesh, endpoint=True)
    tau = np.linspace(0, 1, 20, endpoint=True)

    Theta_array = np.zeros_like(X) # non-dim temp array
    temperature_array = np.zeros_like(X)
    position_array = np.linspace(0, L1 + L2, mesh, endpoint=True)

    # prepare for time marching
    L_char = L1 + L2
    T_final = (L1*T1 + L2*T2) / L_char
    L = L1 / L_char
    Theta_1 = (T1 - T_final) / (T1 - T2)
    Theta_2 = (T2 - T_final) / (T1 - T2)


    A0 = (Theta_1 - Theta_2) * L + Theta_2
    for j in range(len(tau)):
        for i in range(len(X)):
            summation = get_summation(Theta_1, Theta_2, L, X[i], tau[j], 1000)
            Theta_array[i] = A0 + summation
        # make dimensional
        temperature_array = Theta_array * (T1 - T2) + T_final
        plt.plot(position_array, temperature_array)

    plt.xlabel('Position',fontsize=16)
    plt.ylabel('Temperature',fontsize=16)
    plt.title('Temperature vs Position',fontsize=16)

    if save_fig: plt.savefig('hw5p4g_1.png')
    if show_fig: plt.show()

    plt.close()
    return

if __name__ == "__main__":
    problem_4(show_fig=False, save_fig=True)





