import numpy as np
import matplotlib.pyplot as plt

def calculate_surface_temperature(a_L, a_T_c, a_h_c, a_T_inf, a_h_inf, a_k, a_D):
    m = np.sqrt(4 * a_h_c / (a_k * a_L))
    n = np.sqrt(4 * a_h_inf / (a_k * a_D))
    c_0 = m * np.sinh(m * a_L) / np.cosh(m * a_L)
    c_1 = -1 * n * np.sinh(n * a_L) / np.cosh(n * a_L)
    T_s = 1 * (c_0 * a_T_c - c_1 * a_T_inf) / (c_0 - c_1)
    return T_s

def calculate_exact_solution(a_L, a_T_c, a_h_c, a_T_inf, a_h_inf, a_k, a_D, a_T_s, a_N):
    lower_domain = np.linspace(-a_L, 0, a_N, endpoint=True)
    upper_domain = np.linspace(0, a_L, a_N, endpoint=True)
    lower_T = np.zeros_like(lower_domain)
    upper_T = np.zeros_like(upper_domain)

    m = np.sqrt(4 * a_h_c / (a_k * a_L))
    n = np.sqrt(4 * a_h_inf / (a_k * a_D))

    c_lower = (a_T_s - a_T_c) / (2 * np.cosh(m * a_L))
    c_upper = (a_T_s - a_T_inf) / (2 * np.cosh(n * a_L))

    for i in range(len(lower_domain)):
        lower_T[i] = c_lower * (np.exp(m * (a_L + lower_domain[i])) + np.exp(m * (a_L + lower_domain[i]))) + a_T_c
        upper_T[i] = c_upper * (np.exp(n* (upper_domain[i] - a_L)) + np.exp(n* (-upper_domain[i] + a_L))) + a_T_inf

    T = np.append(lower_T[:-1], upper_T) # prevents duplicate point
    X = np.append(lower_domain[:-1], upper_domain)
    return T, X

def calculate_numerical_solution(a_L, a_T_c, a_h_c, a_T_inf, a_h_inf, a_k, a_D, a_T_s, a_N):
    # coffee side
    A = np.zeros([a_N, a_N])
    b = np.zeros(a_N)
    dx = a_L/a_N
    lower_domain = np.linspace(-a_L, 0, a_N, endpoint=True)
    upper_domain = np.linspace(0, a_L, a_N, endpoint=True)

    # assemble lower domain matrix and solve for simplicity
    c = (4 * a_h_c * dx ** 2) / (a_D * a_k)
    t_i_c = 2 + c
    for i in range(a_N):
        if i == 0:
            A[i, i] = 1
            A[i, i + 1] = -1
            b[i] = 0
        elif i == a_N - 1:
            A[i, i] = 1
            b[i] = a_T_s
        else:
            A[i, i] = -t_i_c
            A[i, i - 1] = 1
            A[i, i + 1] = 1
            b[i] = -c*a_T_c
    print(A, b)
    T_lower = np.linalg.solve(A, b)
    print(lower_domain)
    plt.plot(lower_domain, T_lower)

    A = np.zeros([a_N, a_N])
    b = np.zeros(a_N)
    # assemble upper domain matrix and solve for simplicity
    c = (4 * a_h_inf * dx ** 2) / (a_D * a_k)
    t_i_c = 2 + c
    i = 0

    for i in range(a_N):
        if i == 0:
            A[i, i] = 1
            b[i] = a_T_s
        elif i == a_N - 1:
            A[i, i] = 1
            A[i, i - 1] = -1
            b[i] = 0
        else:
            A[i, i] = -t_i_c
            A[i, i - 1] = 1
            A[i, i + 1] = 1
            b[i] = -c*a_T_inf
    T_upper = np.linalg.solve(A,b)
    X = np.append(lower_domain[:-1],upper_domain)
    T = np.append(T_lower[:-1], T_upper)

    return T, X


def main(exact_solution=True, numerical_solution=True, save_figure=False, show_figure=True):
    # Define physical problem
    L = 0.05
    T_c = 100.0
    h_c = 1000.0
    T_inf = 20.0
    h_inf = 100.0
    k = 15.0
    D = 0.005
    T_s =  calculate_surface_temperature(L, T_c, h_c, T_inf, h_inf, k, D)
    print(T_s)
    # Define numerical problem
    N = 50
    [T_ext, X_ext] = calculate_exact_solution(L, T_c, h_c, T_inf, h_inf, k, D, T_s, N)
    # print(len(X))
    # numerical solution
    [T_num, X_num] = calculate_numerical_solution(L, T_c, h_c, T_inf, h_inf, k, D, T_s, N)

    legend = []
    if exact_solution == True:
        plt.plot(X_ext, T_ext)
        legend.append('exact solution')
    if numerical_solution == True:
        plt.plot(X_num, T_num)
        legend.append('numerical solution')

    plt.xlabel('position [m]')
    plt.ylabel('temperature [deg C]')
    plt.title('Temperature of spoon in coffe cup')
    plt.grid()
    
    plt.legend(legend)

    if show_figure == True:
        plt.show()

    if save_figure == True:
        plt.savefig('hw2p3.png')



if __name__ == '__main__':
    main(save_figure=True, show_figure=True, numerical_solution=True, exact_solution=False)