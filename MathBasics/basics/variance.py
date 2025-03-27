import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

X = np.array([8, 0 ,4 ,5, -3, -5])

N = 1000
X = np.random.randint(1, high=20, size=N)

variance_biased = sum((X - X.mean()) ** 2) / len(X)
variance_unbiased = sum((X - X.mean()) ** 2) / (len(X) - 1)


print("Biased variance: ", variance_unbiased)
print(variance_biased == np.var(X))

print("Unbiased variance: ", variance_unbiased)
print(variance_unbiased == np.var(X, ddof=1))


# # 绘制正态分布
# z_scores = (X - X.mean()) / X.std()
# print("标准差后的数据")


# # 创建图形
# plt.figure(figsize=(10, 6))

# # 绘制标准正态分布曲线
# x = np.linspace(-3, 3, 1000)
# plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, label='Standard Deviation')

# # 绘制标准化后的数据的直方图
# plt.hist(z_scores, bins='auto', density=True, alpha=0.6, color='skyblue', label='Standardized Histogram')

# # 使用核密度估计绘制标准化数据的分布
# if len(z_scores) > 1:
#     density = stats.gaussian_kde(z_scores)
#     plt.plot(x, density(x), 'g-', lw=2, label='Standardized Density')

# # 标记零点（标准正态分布的均值）
# plt.axvline(x=0, color='k', linestyle='--', alpha=0.8, label='Mean (0)')

# plt.title('Standardized Graph')
# plt.xlabel('(Z-score)')
# plt.ylabel('P_density')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.show()