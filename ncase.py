import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from numpy import linalg as LA
from math import sqrt
from scipy.stats import t

N = 12
n = 40  # количество элементов
sigma = sqrt(2.4)
sigma2 = 2.4
tetta_0 = (-1) ** N * N
tetta_1 = 6
tetta_2 = 5
tetta_3 = -0.07
tetta = [tetta_0, tetta_1, tetta_2, tetta_3]
m = 3
mu = 0
E = np.random.normal(mu, sigma, n)
k = np.arange(1, 41, 1, dtype=float)
X = -4 + k * 8 / n

Y = tetta_0 + tetta_1 * X + tetta_2 * (X ** 2) + tetta_3 * (X ** 3) + E  # Рассматривается модель наблюдений
Y_E = tetta_0 + tetta_1 * X + tetta_2 * (X ** 2) + tetta_3 * (X ** 3) + E

def Y_mod(tetta=[]):
    Y_t = tetta[0] + tetta[1] * X + tetta[2] * (X ** 2) + tetta[3] * (X ** 3)  # Модель полезного сигнала имеет вид:
    return Y_t


Y_t = Y_mod(tetta)

X_1 = np.column_stack((np.ones(40, dtype=float), X))
tetta_hat_1 = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ Y
E_hat_1 = Y - X_1 @ tetta_hat_1
alpha_1 = np.linalg.inv(X_1.T @ X_1)[1][1]
T_Y1 = (tetta_hat_1[1]) / ((np.sqrt(alpha_1) * LA.norm(E_hat_1))) * np.sqrt(38)
print(abs(T_Y1))

X_2 = np.column_stack((np.ones(40, dtype=float), X, X ** 2))
tetta_hat_2 = np.linalg.inv(X_2.T @ X_2) @ X_2.T @ Y
E_hat_2 = Y - X_2 @ tetta_hat_2
alpha_2 = np.linalg.inv(X_2.T @ X_2)[2][2]
T_Y2 = tetta_hat_2[2] / ((np.sqrt(alpha_2) * LA.norm(E_hat_2))) * np.sqrt(37)
print(abs(T_Y2))

X_3 = np.column_stack((np.ones(40, dtype=float), X, X ** 2, X ** 3))
tetta_hat_3 = np.linalg.inv(X_3.T @ X_3) @ X_3.T @ Y
E_hat_3 = Y - X_3 @ tetta_hat_3
alpha_3 = np.linalg.inv(X_3.T @ X_3)[3][3]
T_Y3 = (tetta_hat_3[3]) / ((np.sqrt(alpha_3) * LA.norm(E_hat_3))) * np.sqrt(36)
print(abs(T_Y3))

X_4 = np.column_stack((np.ones(40, dtype=float), X, X ** 2, X ** 3, X ** 4))
tetta_hat_4 = np.linalg.inv(X_4.T @ X_4) @ X_4.T @ Y
E_hat_4 = Y - X_4 @ tetta_hat_4
alpha_4 = np.linalg.inv(X_4.T @ X_4)[4][4]
T_Y4 = (tetta_hat_4[4]) / ((np.sqrt(alpha_4) * LA.norm(E_hat_4))) * np.sqrt(35)
print(abs(T_Y4))

X_5 = np.column_stack((np.ones(40, dtype=float), X, X ** 2, X ** 3, X ** 4, X ** 5))
tetta_hat_5 = np.linalg.inv(X_5.T @ X_5) @ X_5.T @ Y
E_hat_5 = Y - X_5 @ tetta_hat_5
alpha_5 = np.linalg.inv(X_5.T @ X_5)[5][5]
T_Y5 = (tetta_hat_5[4]) / ((np.sqrt(alpha_5) * LA.norm(E_hat_5))) * np.sqrt(35)
print(abs(T_Y5))

Norm_E_hat_3 = LA.norm(E_hat_3)
alpha = np.diagonal(np.linalg.inv(X_3.T @ X_3))

print('Для альфа = 0.95 t_0975(37) = 2.03', '\n')

interv = (np.sqrt(alpha / 36) * Norm_E_hat_3)
t_left_95, t_right_95 = t.interval(0.95, n - m)
left_95 = interv * t_left_95 + tetta_hat_3
right_95 = interv * t_right_95 + tetta_hat_3
print(np.round(left_95, 5), np.round(right_95, 5))

print('Для альфа = 0.99 t_0995(37) = 2.72', '\n')

t_left_99, t_right_99 = t.interval(0.99, n - m)
left_99 = interv * t_left_99 + tetta_hat_3
right_99 = interv * t_right_99 + tetta_hat_3
print(np.round(left_99, 5), np.round(right_99, 5))

alpha_t = np.array([])
for k in range(n):
    alpha_t = np.append(alpha_t, X_1[k] @ np.linalg.inv(X_1.T @ X_1) @ X_1[k].T)
phi_h = np.array([])
int_l = np.array([])
int_r = np.array([])

Y_H = tetta_hat_3[0] + tetta_hat_3[1] * X + tetta_hat_3[2] * (X ** 2) + tetta_hat_3[3] * (X ** 3)

phi_h = tetta_hat_3[0] + tetta_hat_3[1] * X + tetta_hat_3[2] * (X ** 2) + tetta_hat_3[3] * (X ** 3)

int_l = phi_h - 2.03 * (Norm_E_hat_3 * (np.sqrt(alpha_t))) / (37 ** (1 / 2))
int_r = phi_h + 2.03 * (Norm_E_hat_3 * (np.sqrt(alpha_t))) / (37 ** (1 / 2))

plt.plot(X, int_r, label='доверительный интервал')
plt.plot(X, int_l, label='доверительный интервал')
plt.plot(X, Y_H, 'r', label='полезный сигнал')
plt.title("для альфа = 0.95 t_0975(37) = 2.03")
plt.ylim([9,25])
plt.xlim([-2,2])
plt.legend()
plt.show()

int_l_1 = phi_h - 2.72 * (Norm_E_hat_3 * (np.sqrt(alpha_t))) / (37 ** (1 / 2))
int_r_1 = phi_h + 2.72 * (Norm_E_hat_3 * (np.sqrt(alpha_t))) / (37 ** (1 / 2))

plt.plot(X, int_r_1, label='доверительный интервал')
plt.plot(X, int_l_1, label='доверительный интервал')
plt.plot(X, Y_H, 'r', label='полезный сигнал')
plt.title("для альфа = 0.99 t_0995(37) = 2.72")
plt.ylim([9,25])
plt.xlim([-2,2])
plt.legend()
plt.show()

plt.plot(X, Y_H, 'r', label='полезный сигнал')
plt.plot(X, Y_E, 'r', label='модель наблюдений')
plt.plot(X, Y_t, 'b', label='модель полезного сигнала')
plt.plot(X, int_r, label='доверительный интервал')
plt.plot(X, int_l, label='доверительный интервал')
plt.ylim([9,25])
plt.xlim([-2,2])
plt.legend()
plt.show()

#fig, axes = plt.subplots(1, 2)
#sns.histplot(E_hat_3, bins=6, common_norm=True, ax=axes[0])
#sns.kdeplot(E_hat_3, fill=True, ax=axes[1])

sns.distplot(E_hat_3, hist=True, kde=True,
    bins=6, color = 'darkblue',
    hist_kws={'edgecolor':'black'},
    kde_kws={'linewidth': 4})
plt.show()

sigma_hat_3_sq = 1 / n * Norm_E_hat_3 ** 2
print(sigma_hat_3_sq)

l = 6
step = (E_hat_3.max() - E_hat_3.min()) / (l)
prev = E_hat_3.min()
edge = prev + step
n_vec = []
t = [prev]
for i in range(l - 1):
    print(((prev <= E_hat_3) & (E_hat_3 <= edge)).sum())
    n_vec.append(((prev <= E_hat_3) & (E_hat_3 <= edge)).sum())
    t.append(edge)
    prev = edge
    edge += step

t = np.array(t)
n_vec = np.array(n_vec)
p_hat = n_vec / n
p = []
for i in range(l - 1):
    p.append(sp.stats.norm.cdf(t[i + 1] / sigma_hat_3_sq) - sp.stats.norm.cdf(t[i] / sigma_hat_3_sq))
p = np.array(p)
T_Z = (((p - p_hat) ** 2) / p).sum() * n
print(T_Z)
