import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def ComputeLoss(X, y, theta):
    m = y.shape[1]
    J = np.dot(theta.T, X)-y
    J = np.sum(J**2)
    return J/(2*m)


def gradientDescent(X, y, theta, alpha, num_iters):
    res = theta
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        dt = np.mean((np.dot(res.T, X)-y)*X, axis=1, keepdims=True)
        res -= alpha*dt
        J_history[iter] = ComputeLoss(X, y, theta=res)
        print(f'Epoch [{iter+1}/{num_iters}]: Train loss: {J_history[iter]:.4f}')
    return res, J_history


train_data = pd.read_table(r"Lab1\\ex1data1.txt", sep=",", header=None)

x = np.array(train_data.iloc[:, 0])
y = np.array(train_data.iloc[:, 1]).reshape(1, -1)

plt.figure(1)
plt.scatter(x, y, marker='x', c='r')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')

theta = np.zeros((2, 1))
x = np.array([x, np.ones(x.shape)])

# Hyperparameter
iterations = 3000
alpha = 0.008


# loss = ComputeLoss(x,y,theta=theta)
# print(loss)
theta, J = gradientDescent(x, y, theta=theta, alpha=alpha, num_iters=iterations)
print(f'Theta found by gradient descent: {theta[0,0]:.3f} {theta[1,0]:.3f}')

# plot ans line
ansx = np.linspace(0, 30, 200)
ansy = theta[0, 0]*ansx+theta[1, 0]
plt.plot(ansx, ansy, 'b-')

# plot loss fuction
plt.figure(2)
plt.plot(J)
plt.title('Loss Fuction')
plt.xlabel('epoch')

# plot loss space
plt.figure(3)
ax3 = plt.axes(projection='3d')
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_val = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

#  Fill out J_vals
for i in range(theta0_vals.shape[0]):
    for j in range(theta1_vals.shape[0]):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_val[i, j] = ComputeLoss(x, y, t)
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
ax3.plot_surface(theta0_vals, theta1_vals, J_val.T, cmap='rainbow')


plt.show()
