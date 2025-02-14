import numpy as np

rho = 0.921
c   = 1.012e3
k   = 0.032
mu  = 2.223e-5
nu  = mu / rho
Pr  = 0.704
g   = 9.81
alpha = k/rho/c
print(alpha)

T_s = 200
T_inf = 20
T_r = 0.5*(T_s+T_inf) + 273.15

L_c_vert = 1
L_c_horz = 1/4
def get_Ra(L_c):
    return g * (T_s - T_inf) * L_c**3 / (T_r * nu * alpha)

Ra_vert = get_Ra(L_c_vert)
Ra_hori = get_Ra(L_c_horz)

Nu_v = (0.825 + 0.387 * Ra_vert**(1/6) / (1 + (0.492/Pr)**(9/16))**(8/27))**2
Nu_h = 0.15 * Ra_hori**(1/3)



print('Ra_v = {}'.format(Ra_vert))
print('Ra_h = {}'.format(Ra_hori))

print('Nu_v = {}'.format(Nu_v))
print('Nu_h = {}'.format(Nu_h))
print(0.52 * Ra_hori**(0.2))