import numpy as np
import matplotlib.pyplot as plt

r_i = 0.006
r_o = 0.008
k_ins = 0.05
k = 385
h = 5

r_e = np.linspace(0.008,0.018,100,endpoint=True)
r_e_critical = k_ins/h
print(r_e_critical)

equivalent_resistance = 1/(2*np.pi*k)*np.log(r_o/r_i) + 1/(2*np.pi*k_ins)*np.log(r_e/r_o) + 1/(2*np.pi*r_e*h)
conductive_resistance = 1/(2*np.pi*k)*np.log(r_o/r_i) + 1/(2*np.pi*k_ins)*np.log(r_e/r_o)
convective_resistance = 1/(2*np.pi*r_e*h)
critical_resistance = 1/(2*np.pi*k)*np.log(r_o/r_i) + 1/(2*np.pi*k_ins)*np.log(r_e_critical/r_o)\
                      + 1/(2*np.pi*r_e_critical*h)

r = r_e - r_o
r_critical = r_e_critical - r_o
print(r_critical)
labels = []
plt.plot(r, equivalent_resistance)
labels.append('Equivalent Resistance')
plt.plot(r, conductive_resistance)
labels.append('Conductive Resistance')
plt.plot(r, convective_resistance)
labels.append('Convective Resistance')
plt.plot(r_critical, critical_resistance, 'ro')
labels.append('Critical Resistance')
plt.legend(labels)
plt.xlabel('r = r_e - r_o',fontweight='bold', fontsize=14)
plt.ylabel('Resistance',fontweight='bold', fontsize=16)
plt.title('Problem 2 Part F', fontweight='bold', fontsize=16)
plt.savefig('hw2p2f.png')
plt.show()

