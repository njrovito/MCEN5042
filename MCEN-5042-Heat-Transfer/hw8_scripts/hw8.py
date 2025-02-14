import numpy as np
import matplotlib.pyplot as plt


def linear_interpolate(x, prop_0, prop_1):
    prop_out = np.zeros(len(prop_0))
    prop_out[0] = x
    for i in range(1, len(prop_out)):
        prop_out[i] = (prop_1[i] - prop_0[i])/(prop_1[0] - prop_0[0]) * (prop_out[0] - prop_1[0]) + prop_1[i]
    return prop_out

def calculate_mass_rate(a_u_m, a_D, a_rho):
    mass_flow_rate = a_rho * a_u_m * (np.pi * a_D**2 / 4)
    return mass_flow_rate

def calculate_Reylonds_number(a_Lc, a_mu, a_u, a_rho):
    return a_Lc * a_u * a_rho / a_mu

def get_entry_lengths(a_Re, a_Pr, a_D):
    if a_Re > 10e5:
        x_h = -50 * a_D
        x_t = 10 * D
    else:
        x_h = 0.05 * a_Re * a_D
        x_t = 0.05 * a_Re * a_Pr

    return x_h, x_t

def calculate_Nusselt_number(a_Re, a_Pr, a_L, a_D, a_visc_ratio):
    return 1.86 * (a_Re * a_Pr * a_D / a_L)**(1/3) * a_visc_ratio**0.14

def calculate_h_bar(Nu, k, D):
        return Nu * k / D

def T_mean(z, Ts, Ti, h_bar, rho, cp, um, D):
    R = D/2
    return Ts - (Ts - Ti) * np.exp(2 * h_bar * z / (rho * cp * um * R))


def main():
    T_i = 50 + 273.15
    T_s = 70 + 273.15
    T_avg = 0.5 * (T_s + T_i)
    T_avg = 61.40 + 217.15


    prop_T_330 = [330, 4.184e3, 489.0e-6, 650e-3, 3.15]
    prop_T_335 = [335, 4.186e3, 453.0e-6, 656e-3, 2.88]
    prop_T_340 = [340, 4.188e3, 420.0e-6, 660e-3, 2.66]
    prop_T_345 = [345, 4.191e3, 389.0e-6, 668e-3, 2.45]

    rho = 983.13  # [kg/m^3]
    D   = 0.0524  # [m]
    L   = 3.0     # [m]
    u_m = 0.02    # [m/s]

    # part a
    T, c, mu, k, Pr = linear_interpolate(T_avg, prop_T_330, prop_T_335)
    # T_mean = 62 + 273.15
    print('Part a: ')
    print('-' * 50)
    print('T_eval = {}'.format(T))
    print()

    # part b
    print('Part b: ')
    print('-' * 50)
    print('rho = {}, c_p = {}, mu = {}, k = {}, Pr = {}'.format(rho, c, mu, k, Pr))
    print()

    # part c
    m_dot = calculate_mass_rate(u_m, D, rho)
    Re = calculate_Reylonds_number(a_Lc=D, a_mu=mu, a_u=u_m, a_rho=rho)
    x_h, x_t = get_entry_lengths(Re, Pr, D)
    print('Part c: ')
    print('-' * 50)
    print('mass flow rate = {}'.format(m_dot))
    print('Re = {}'.format(Re))
    print('hydrodynamic entry length = {}'.format(x_h))
    print('thermodynamic entry length = {}'.format(x_t))
    print()

    # part d
    T_s, c_s, mu_s, k_s, Pr_s = linear_interpolate(T_s, prop_T_340, prop_T_345)
    viscosity_ratio = mu / mu_s
    Pr_check = False
    mu_check = False
    if 0.6 <= Pr <= 5.0: Pr_check = True
    if 0.0044 <= np.abs(viscosity_ratio) <= 9.75: mu_check = True
    Nu = calculate_Nusselt_number(Re, Pr, L, D, viscosity_ratio)
    print('Part d: ')
    print('-' * 50)
    print('Pr = {}, in range: {}'.format(Pr, Pr_check))
    print('Visc ratio = {}, in range: {}'. format(viscosity_ratio, mu_check))
    print('Nu = {}'.format(Nu))
    print()

    print('Part e: ')
    print('-' * 50)
    h_bar = calculate_h_bar(Nu, k, D)
    Tm0 = T_mean(0, T_s, T_i, h_bar, rho, c, u_m, D)
    Tml = T_mean(L, T_s, T_i, h_bar, rho, c, u_m, D)

    print(Tm0, Tml)
    def mean_log_temp(ts, tml, tm0):
        return ((ts - tml) - (ts-tm0)) / (np.log((ts-tml)/(ts-tm0)))

    mlt = mean_log_temp(T_s, Tml, Tm0)
    qs1 = D * np.pi * L * h_bar * mlt
    qs2 = m_dot * c * (Tml - Tm0)
    print(qs1 + qs2)



if __name__ == '__main__':
    main()