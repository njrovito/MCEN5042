import numpy as np
import matplotlib.pyplot as plt


def get_alpha(A_inputs):
    T_A, k_A, rho_A, c_A = A_inputs
    alpha = k_A / (rho_A * c_A)
    return(alpha)






def main():
    # inputs = [T0, k, rho, c_p]
    skin_inputs = [37, 0.4, 1000, 3500]
    T_A = 20 # just needs to be colder than T_skin
    copper_inputs = [T_A, 380, 8800, 400]
    glass_inputs = [T_A, 1.4, 2500, 750]

    alpha_copper = get_alpha(copper_inputs) * 380
    alpha_glass = get_alpha(glass_inputs) * 1.4
    print(alpha_copper, alpha_glass)


if __name__ == '__main__':
    main()