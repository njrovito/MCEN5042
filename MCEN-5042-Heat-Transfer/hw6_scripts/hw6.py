import numpy as np
import matplotlib.pyplot as plt

def problem_1_b(rho, u_c, L_c, mu, plot_name, tripped=False):
    Re_c = 5e5
    N = 100

    Re = rho * u_c * L_c / mu
    x_c = Re_c * mu / rho / u_c

    x_pos = np.linspace(0, L_c, N)
    delta = np.zeros_like(x_pos)

    for i in range(N):
        if x_pos[i] <= x_c:
            delta[i] = 5 * x_pos[i] * Re**(-0.5)
        else:
            delta[i] = 0.37 * x_pos[i] * Re**(-0.2)
    print('Reynolds: {}, Critical x: {}'.format(Re, x_c))

    if tripped:
        for i in range(N):
            delta[i] = 0.37 * x_pos[i] * Re ** (-0.2)

    plt.plot(x_pos, delta)
    plt.ylim([0.0, 0.025])
    plt.xlabel('position [m]')
    plt.ylabel('boundary thickness [m]')
    plt.title(plot_name[:-4])
    plt.savefig(plot_name)
    plt.close()

    return x_c

def problem_1_c(x_c, L_c, u_inf, nu, k, Pr, tripped=False):
    N = 100
    x_pos = np.linspace(0, L_c, N)
    h = np.zeros_like(x_pos)
    h_lam = np.zeros_like(x_pos)
    h_turb = np.zeros_like(x_pos)
    h_bar = np.zeros_like(x_pos)

    C_lam = 0.332 * (u_inf / nu)**0.5 * k * Pr**(1/3)
    C_turb = 0.0296 * (u_inf / nu)**0.8 * k * Pr**(1/3)

    # print(C_lam, C_turb)

    for i in range(N):
        h_lam[i] =  C_lam / (x_pos[i]**0.5)
        h_turb[i] = C_turb / (x_pos[i]**0.2)
        h_bar[i] = 0
        if x_pos[i] <= x_c:
            h[i] = C_lam / (x_pos[i]**0.5)
            h_bar[i] = C_lam / (0.5 * L_c) * x_c ** 0.5
        elif tripped:
            h[i] = C_turb / (x_pos[i] ** 0.2)
            h_bar[i] += C_turb / (0.8 * L_c) * (L_c ** 0.8 - x_c ** 0.8)
        else:
            h[i] = C_turb / (x_pos[i]**0.2)
            h_bar[i] += C_turb / (0.8 * L_c) * (L_c**0.8 - x_c**0.8)
    plt.plot(x_pos, h)
    plt.plot(x_pos, h_lam)
    plt.plot(x_pos, h_turb)
    plt.plot(x_pos, h_bar)
    legend = ['h', 'h_lam', 'h_turb', 'h_bar']
    plt.legend(legend)
    plt.show()
    plt.close()
    return




def problem_1():
    length = 1
    temp_surface = 400
    temp_farfield = 300
    vel_farfield = 1

    ## AIR PROPERTIES ##
    rho_air = 0.9950
    cp_air = 1.009
    mu_air = 208.2e-7
    k_air = 30.0e-3
    Pr_air = 0.690
    nu_air = mu_air / rho_air
    alpha_air = k_air / rho_air / cp_air
    print('Nu, air = {}:     alpha, air = {}'.format(nu_air, alpha_air))

    ## WATER PROPERTIES ##
    rho_water = 973.7
    cp_water = 4.256
    mu_water = 217e-6
    k_water = 688e-3
    Pr_water = 1.34
    nu_water = mu_water / rho_water
    alpha_water = k_water / rho_water / cp_water
    print('Nu, water = {}:     alpha, air = {}'.format(nu_water, alpha_water))

    x_c = problem_1_b(rho_air, vel_farfield, length, mu_air, 'air.png')
    problem_1_c(x_c, length, vel_farfield, nu_air, k_air, Pr_air)

    problem_1_b(rho_water, vel_farfield, length, mu_water, 'water.png')
    problem_1_c(x_c, length, vel_farfield, nu_water, k_water, Pr_water)

    problem_1_b(rho_water, vel_farfield, length, mu_water, 'tripped_water.png', tripped=True)
    problem_1_c(x_c, length, vel_farfield, nu_water, k_water, Pr_water, tripped=True)

def main():
    problem_1()


if __name__ == '__main__':
    main()
