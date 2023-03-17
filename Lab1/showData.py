import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_table(r"Lab1\\machine-learning-ex1\\ex1\\ex1data1.txt",sep = ",",header=None)

x = data.iloc[:,0]
y = data.iloc[:,1]

print(x.dtype)
plt.scatter(x,y,marker = 'x')
plt.ylabel('Profit in $10,000s');
plt.xlabel('Population of City in 10,000s');
plt.show()