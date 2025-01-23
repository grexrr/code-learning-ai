import numpy as np

# 定义模型的预测函数
def model(x, theta):
    return theta[0] * x + theta[1]

# 定义损失函数
def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 定义损失函数对参数的梯度
# back propagation is essentially propagating the gradient of the loss function to the parameters
def gradient(x, y_true, theta):
    n = len(x)
    y_pred = model(x, theta)
    d_theta0 = -2 / n * np.sum(x * (y_true - y_pred))
    d_theta1 = -2 / n * np.sum(y_true - y_pred)
    return np.array([d_theta0, d_theta1])

# 梯度下降迭代
def gradient_decent(x, y, theta, learning_rate, iterations):
    for i in range(iterations):
        grad = gradient(x, y, theta)
        theta -= learning_rate * grad
        if i % 10 == 0:
            current_loss = loss(y, model(x, theta))
            print(f"Iteration {i}, Loss: {current_loss}, Theta: {theta}")
    return theta

# 测试
np.random.seed(42)
x = np.linspace(0, 10, 100)
true_theta = [2, 1]
y = model(x, true_theta) + np.random.normal(scale=1, size=len(x))

# 初始化参数
initial_theta = np.random.randn(2)
learning_rate = 0.01
iterations = 100

# 运行梯度下降
optimized_theta = gradient_decent(x, y, initial_theta, learning_rate, iterations)
print(f"Optimized Theta: {optimized_theta}")