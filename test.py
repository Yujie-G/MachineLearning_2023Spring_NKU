import numpy as np

test_data = np.array([1650,3]).T
test_data = (test_data-1)/2
theta = np.array([[2,1,3]])
ansy = theta.dot(np.append(test_data,1))
print(ansy[0])

