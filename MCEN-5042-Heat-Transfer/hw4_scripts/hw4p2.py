import numpy as np
import matplotlib.pyplot as plt

def problem_2(a_inputs, W=None):
    d = 2 * a_inputs[0]
    D = 2 * a_inputs[1]
    L = a_inputs[2]
    k = a_inputs[3]
    T_1 = a_inputs[4]
    T_2 = a_inputs[5]

    # part b
    W = np.linspace(0.0, np.sqrt(-0.25 * (2 * D * d - D**2 - d**2)) / 1.01, 50)
    q = np.zeros_like(W)

    numerator = 2 * np.pi * L * k * (T_1 - T_2)

    for i in range(len(W)):
        denominator = np.arccosh((D**2 + d**2 -4 * W[i]**2) / (2*D*d))
        q[i] = numerator / denominator

    # part c
    W = 0.003
    denominator = np.arccosh((D**2 + d**2 -4 * W**2) / (2*D*d))
    q = numerator / denominator
    print('The heat transfer rate for part c: {}'.format(q))

    # part d
    W = 0.00
    denominator = np.arccosh((D ** 2 + d ** 2 - 4 * W ** 2) / (2 * D * d))
    q = numerator / denominator
    print('The heat transfer rate for part d: {}'.format(q))

if __name__ == '__main__':
    # inputs = [r1, r2, L, k, T1, T2]
    inputs = [0.001, 0.005, 0.10, 0.14, 150, 40]

    problem_2(a_inputs=inputs)