import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import math

Steps = 300  # Количество шагов анимации

t = np.linspace(0, 7, Steps)  # Время от 0 до 7 с 300 шагами

# Параметры системы
PlateHeight = 4  # Высота пластины
g = 9.81  # Ускорение свободного падения
m = 1  # Масса
J = 3  # Момент инерции
alpha = np.pi / 6  # Угол наклона пластины
PlateWidth = PlateHeight / np.tan(alpha)  # Ширина пластины
k = 10  # Коэффициент упругости
c = 10  # Коэффициент демпфирования

# Решение системы дифференциальных уравнений методом Крамера
def odesys(y, t, g, m, J, alpha, k, c):
    # y: [s, phi, s', phi']
    dy = np.zeros(4)
    dy[0] = y[2]  # ds/dt = s'
    dy[1] = y[3]  # dphi/dt = phi'

    # Коэффициенты системы уравнений
    a11 = J + m * y[0] ** 2 * np.sin(alpha) ** 2
    a12 = 0
    a21 = 0
    a22 = m

    # Правые части уравнений
    b1 = -c * y[1] - 2 * m * y[0] * y[2] * y[3] * np.sin(alpha) ** 2
    b2 = -k * y[2] - m * g * np.cos(alpha) + m * y[0] * y[3] ** 2 * np.sin(alpha) ** 2

    # Решение системы методом Крамера
    dy[2] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)  # s''
    dy[3] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)  # phi''

    # dy: [s', phi', s'', phi'']
    return dy

# Начальные условия
s0 = 0  # Начальное смещение
phi0 = math.pi / 6  # Начальный угол поворота
ds0 = 20  # Начальная скорость смещения
dphi0 = 0  # Начальная угловая скорость
y0 = [s0, phi0, ds0, dphi0]

# Решаем систему дифференциальных уравнений
Y = odeint(odesys, y0, t, (g, m, J, alpha, k, c))

# Извлекаем элементы из массива
d = Y[:, 0]  # s(t) - смещение
phi = Y[:, 1]  # phi(t) - угол поворота

# Создание вращающейся пластины и точки в ней
StandZ = 1  # Высота стойки

# Координаты углов пластины
AX = PlateWidth / 2 * np.cos(phi)
AY = PlateWidth / 2 * np.sin(phi)
AZ = StandZ

BX = -PlateWidth / 2 * np.cos(phi)
BY = -PlateWidth / 2 * np.sin(phi)
BZ = StandZ

CX = BX
CY = BY
CZ = BZ + PlateHeight

DX = AX
DY = AY
DZ = AZ + PlateHeight

PathWidth = d * np.cos(alpha)  # Ширина траектории движения точки

# Абсолютные координаты точки
pointZ = StandZ + PlateHeight / 2 + d * np.sin(alpha)  # Высота точки
pointX = PathWidth * np.cos(phi)  # Координата X точки
pointY = PathWidth * np.sin(phi)  # Координата Y точки

# Создание графиков физических величин
fig2 = plt.figure()
ax2 = fig2.add_subplot(2, 2, 1)
ax2.plot(Y[:, 0])
ax2.set_title("s")  # График смещения s(t)

ax4 = fig2.add_subplot(2, 2, 2)
ax4.plot(Y[:, 2])
ax4.set_xlim(left=0, right=100)
ax4.set_title("s'")  # График скорости смещения s'(t)

ax3 = fig2.add_subplot(2, 2, 3)
ax3.plot(Y[:, 1])
ax3.set_title("phi")  # График угла поворота phi(t)

ax5 = fig2.add_subplot(2, 2, 4)
ax5.plot(Y[:, 3])
ax5.set_title("phi'")  # График угловой скорости phi'(t)
ax5.set_xlim(left=0, right=100)

# Создание 3D-анимации
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set(xlim=[-8, 8], ylim=[-8, 8], zlim=[0, 6])  # Устанавливаем пределы осей

# Начальное положение точки и пластины
pointPlot, = ax.plot([pointX[0]], [pointY[0]], [pointZ[0]], marker='o', markersize='3')
lineABPLOT, = ax.plot([AX[0], BX[0]], [AY[0], BY[0]], [AZ, BZ], color='black', linewidth='4')
lineCDPLOT, = ax.plot([CX[0], DX[0]], [CY[0], DY[0]], [CZ, DZ], color='black', linewidth='4')
lineADPLOT, = ax.plot([AX[0], DX[0]], [AY[0], DY[0]], [AZ, DZ], color='black', linewidth='4')
lineBCPLOT, = ax.plot([BX[0], CX[0]], [BY[0], CY[0]], [BZ, CZ], color='black', linewidth='4')
lineBDPLOT, = ax.plot([BX[0], DX[0]], [BY[0], DY[0]], [BZ, DZ], color='black', linewidth='4', alpha=0.3)

# Ось вращения
axis = ax.plot([0, 0], [0, 0], [0, StandZ], color='black', linewidth='2')
axis1 = ax.plot([0, 0], [0, 0], [StandZ + PlateHeight, StandZ + PlateHeight + 1], color='black', linewidth='2')
axis2 = ax.plot([-1, 1], [0, 0], [0, 0], color='black', linewidth='2')

# Функция для анимации
def Anima(i):
    pointPlot.set_data_3d([pointX[i]], [pointY[i]], [pointZ[i]])  # Обновляем координаты точки
    lineABPLOT.set_data_3d([AX[i], BX[i]], [AY[i], BY[i]], [AZ, BZ])  # Обновляем линию AB
    lineCDPLOT.set_data_3d([CX[i], DX[i]], [CY[i], DY[i]], [CZ, DZ])  # Обновляем линию CD
    lineADPLOT.set_data_3d([AX[i], DX[i]], [AY[i], DY[i]], [AZ, DZ])  # Обновляем линию AD
    lineBCPLOT.set_data_3d([BX[i], CX[i]], [BY[i], CY[i]], [BZ, CZ])  # Обновляем линию BC
    lineBDPLOT.set_data_3d([BX[i], DX[i]], [BY[i], DY[i]], [BZ, DZ])  # Обновляем линию BD
    return [pointPlot, lineABPLOT, lineCDPLOT, lineBCPLOT, lineADPLOT, lineBDPLOT]

# Создаем анимацию
anima = FuncAnimation(fig, Anima, frames=Steps, interval=24)

# Отображаем анимацию
plt.show()