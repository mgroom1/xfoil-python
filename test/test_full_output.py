import numpy as np
import matplotlib.pyplot as plt

from xfoil import XFoil
from xfoil.model import Airfoil

file_name = 'naca2412-il.dat'
alpha = 0

x, y = np.genfromtxt(file_name, dtype=float, skip_header = 1).T
airfoil = Airfoil(x=x, y=y)

xf = XFoil()
xf.airfoil = airfoil

xf.Re = 5000000
xf.M = 0.2
xf.n_crit = 9
xf.max_iter = 50

xf.repanel()
xf.filter()

cl2, cd2, _, _, conv2 = xf.a(alpha)
x2, y2, cp2 = xf.get_cp_distribution()

cl, cd, cm, x, y, cp, tau, uedg, delt, dstr, thet, tstr, conv = xf.a_full(alpha)
print(conv, conv2)
print(cl, cl2, cd, cd2)

print(xf.get_section_properties())

fig, ax = plt.subplots()
ax.plot(x2, cp2)
ax.scatter(x, cp)

dx = x[1:] - x[:-1]
dy = y[1:] - y[:-1]
ds = np.hypot(dx, dy)
nx = dy/ds
ny = -dx/ds

Nx = np.zeros(len(x))
Nx[1:-1] = (nx[1:] + nx[:-1])/2
Nx[0] = nx[0]
Nx[-1] = nx[-1]

Ny = np.zeros(len(x))
Ny[1:-1] = (ny[1:] + ny[:-1])/2
Ny[0] = ny[0]
Ny[-1] = ny[-1]

fig, ax = plt.subplots()
ax.plot(x,y,c='k')
#ax.plot(x,cp)
ax.plot(x + Nx*delt, y + Ny*delt)
ax.plot(x + Nx*dstr, y + Ny*dstr)
ax.plot(x + Nx*thet, y + Ny*thet)
ax.plot(x + Nx*tstr, y + Ny*tstr)

cl, cd, cm, tau, UEDG, dueds, delt, dstr, thet, tstr, conv = xf.a_bl_te(alpha)

ax.scatter([x[0] + Nx[0]*delt[0], x[-1] + Nx[-1]*delt[1]],
           [y[0] + Ny[0]*delt[0], y[-1] + Ny[-1]*delt[1]])
ax.scatter([x[0] + Nx[0]*dstr[0], x[-1] + Nx[-1]*dstr[1]],
           [y[0] + Ny[0]*dstr[0], y[-1] + Ny[-1]*dstr[1]])
ax.scatter([x[0] + Nx[0]*thet[0], x[-1] + Nx[-1]*thet[1]],
           [y[0] + Ny[0]*thet[0], y[-1] + Ny[-1]*thet[1]])
ax.scatter([x[0] + Nx[0]*tstr[0], x[-1] + Nx[-1]*tstr[1]],
           [y[0] + Ny[0]*tstr[0], y[-1] + Ny[-1]*tstr[1]])

ax.set_aspect('equal')

# plot edge velocity
fig, ax = plt.subplots()
ax.plot(x, uedg)
ax.scatter([x[0], x[-1]], [UEDG[0], UEDG[1]])

# compare dueds
xm = (x[1:]+x[:-1])/2
N = len(xm)//2
ds = np.hypot(x[1:]-x[:-1], y[1:]-y[:-1])
dueds_full = -(uedg[1:] - uedg[:-1])/ds
fig, ax = plt.subplots()
ax.plot(xm[:N], dueds_full[:N])
ax.plot(xm[N:], -dueds_full[N:])
ax.scatter([xm[0], xm[-1]], [dueds[0], dueds[1]])

plt.show()


