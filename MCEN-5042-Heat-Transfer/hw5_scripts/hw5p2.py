import numpy as np
import matplotlib.pyplot as plt

def part_a(t_star, T_star, T_inf, T_i):
    tau = -t_star / np.log((T_star - T_inf) / (T_i - T_inf))
    return tau

def part_b(T_inf, T_i, tau, show_plot=False, save_plot=False):
    time = np.linspace(0,1000,100)
    T = np.zeros_like(time)
    const = T_i - T_inf
    i = 0
    for i in range(len(time)):
        T[i] = T_inf + const * np.exp(-time[i] / tau)

    plt.plot(time, T)
    plt.xlabel('time [s]')
    plt.ylabel('temperature')
    plt.title('problem 2 part b')
    plt.grid()

    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig('hw5p2.png')
    return

def part_c(D):  # for sphere
    char_length = D/3
    return char_length

def part_d(rho, c, l_c, tau):
    h = rho * c * l_c / tau
    return h

def part_e(h, l_c, k):
    Bi = h * l_c / k
    return Bi

def main():
    diameter = 12.7e-3
    conductivity = 401
    density = 8933
    heat_capacity = 385
    initial_temp = 66
    inf_temp = 27
    star_temp = 55
    star_time = 69

    # part a
    tau = part_a(star_time, star_temp, inf_temp, initial_temp)
    print('Time constant: {}'.format(tau))

    # part b
    part_b(inf_temp, initial_temp, tau, False, True)

    # part c
    char_length = part_c(diameter)
    print('Characteristic length: {}'.format(char_length))

    # part d
    convection_coef = part_d(density, heat_capacity, char_length, tau)
    print('Convection coefficient: {}'.format(convection_coef))

    # part e
    biot = part_e(convection_coef, char_length, conductivity)
    print('Biot number: {}'.format(biot))


if __name__ == '__main__':
    main()

