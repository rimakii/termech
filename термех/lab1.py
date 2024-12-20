import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.animation import FuncAnimation
import sympy as sp

t = sp.Symbol('t')
T = np.linspace(1, 40, 1000)
R = 4
Omega = 1
r = 1 + sp.sin(5 * t)
phi = t
x = r * sp.cos(phi)
y = r * sp.sin(phi)
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Vmod = sp.sqrt(Vx * Vx + Vy * Vy)
wx = sp.diff(Vx, t)
wy = sp.diff(Vy, t)
wmod = sp.sqrt(wx * wx + wy * wy)
wtau = sp.diff(Vmod, t)
rho = (Vmod * Vmod) / sp.sqrt(wmod * wmod - wtau * wtau)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
Rho = np.zeros_like(T)
Phi = np.zeros_like(T)
for i in np.arange(len(T)):
    X[i] = x.subs(t, T[i])
    Y[i] = y.subs(t, T[i])
    VX[i] = Vx.subs(t, T[i])
    VY[i] = Vy.subs(t, T[i])
    AX[i] = wx.subs(t, T[i])
    AY[i] = wy.subs(t, T[i])
    Rho[i] = rho.subs(t, T[i])
    Phi[i] = phi.subs(t, T[i])

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-4 * R, 4 * R], ylim=[-R, R])
ax1.plot(X, Y)
P, = ax1.plot(X[0], Y[0], marker='o')
Vline, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
Aline, = ax1.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'y')
Tline, = ax1.plot([0, X[0]], [0, Y[0]], 'c')
ArrowX = np.array([-0.2 * R, 0, -0.2 * R])
ArrowY = np.array([0.1 * R, 0, -0.1 * R])
ArrowAX = np.array([-0.2 * R, 0, -0.2 * R])
ArrowAY = np.array([0.1 * R, 0, -0.1 * R])

def anima(j):
    P.set_data([X[j]], [Y[j]])
    Vline.set_data([X[j], X[j] + VX[j]], [Y[j], Y[j] + VY[j]])
    Aline.set_data([X[j], X[j] + AX[j]], [Y[j], Y[j] + AY[j]])
    Tline.set_data([0, X[j]], [0, Y[j]])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[j], VX[j]))
    RArrowAX, RArrowAY = Rot2D(ArrowAX, ArrowAY, math.atan2(AY[j], AX[j]))

    VArrow.set_data(RArrowX + X[j] + VX[j], RArrowY + Y[j] + VY[j])
    AArrow.set_data(RArrowAX + X[j] + AX[j], RArrowAY + Y[j] + AY[j])
    return P, Vline, VArrow, AArrow, Aline, Tline

def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY

RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
RArrowAX, RArrowAY = Rot2D(ArrowAX, ArrowAY, math.atan2(AY[0], AX[0]))
VArrow, = ax1.plot(RArrowX + X[0] + VX[0], RArrowY + Y[0] + VY[0])
AArrow, = ax1.plot(RArrowAX + X[0] + AX[0], RArrowAY + Y[0] + AY[0])
anim = FuncAnimation(fig, anima, frames=1000, interval=30, blit=True)
plt.show()