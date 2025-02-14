import numpy as np
import matplotlib.pyplot as plt

def main():
    N = 500

    T_c = 1202.3
    T1 = 369.1
    T2 = 323.2
    ks = 5.0
    kc = 30.0
    h = 500
    q_dot = 10**5

    R1 = 0.5
    R2 = 0.6

    T = np.zeros(N)
    r = np.linspace(0.0, R2, N, endpoint=True)

    for i in range(N):
        if r[i] < R1:
            T[i] = -q_dot * r[i]**2 / 6 / ks + T_c
        elif r[i] <= R2:
            T[i] = (T2*R1*R2 - T1*R1*R2 + r[i]*(T1*R1 - T2*R2)) / (r[i] * (R1 - R2))

    plt.plot(r,T)
    plt.xlabel('r [m]')
    plt.ylabel('T(r) [k]')
    plt.title('Temperature v Radius, steady sphere with heat gen')
    # plt.savefig('hw3p1e.png')
    plt.show()

if __name__ == "__main__":
    main()