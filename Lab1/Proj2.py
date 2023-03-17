import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def featureNormalize(X):
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, keepdims=True)
    return (X-mu)/sigma, mu, sigma


def ComputeLoss(X, y, theta):
    m = y.shape[1]
    J = np.dot(theta, X)-y
    J = np.sum(J**2)
    return J/(2*m)


def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[1]
    res = theta
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        dt = np.mean((np.dot(res, X)-y)*X, axis=1, keepdims=True).T
        res -= alpha*dt
        J_history[iter] = ComputeLoss(X, y, theta=res)
        print(
            f'Epoch [{iter+1}/{num_iters}]: Train loss: {J_history[iter]:.4f}')
    return res, J_history


train_data = pd.read_table(r"Lab1\\ex1data2.txt", sep=",", header=None)

X = np.array(train_data.iloc[:, 0:2]).T
Y = np.array(train_data.iloc[:, 2]).reshape(1, -1)
len = Y.shape[1]


[X, mu, sigma] = featureNormalize(X)

X = np.row_stack((X, np.ones((1, len))))
theta = np.zeros((1, 3))
print(ComputeLoss(X, Y, theta=theta))

# hyperparameter
alpha = 0.05
iterations = 3000

# train
theta, J = gradientDescent(X, Y, theta=theta, alpha=alpha, num_iters=iterations)
print(f'Theta found by gradient descent: {theta[0,0]:.3f} {theta[0,1]:.3f} {theta[0,2]:.3f}')

# Plot the convergence graph
plt.figure(1)
plt.plot(J, '-b');
plt.title('Loss Fuction')
plt.xlabel('Number of iterations');
plt.ylabel('Cost J');

#test
test_data = np.array([1650,3]).reshape(-1,1)
test_data = (test_data-mu)/sigma
ansy = theta.dot(np.append(test_data,1))
print(f'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):',end='')
print(ansy[0])


plt.show()


# using normal equation
theta_ = np.linalg.inv(X.dot(X.T)).dot(X).dot(Y.T)
print(f'Theta found by normal equation: {theta_[0,0]:.3f} {theta_[1,0]:.3f} {theta_[2,0]:.3f}')