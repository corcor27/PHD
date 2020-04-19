import matplotlib.pyplot as plt
import numpy as np

A = np.array([[2,-1], [-1,2]])

B = np.array([0,3])

inv_A = np.linalg.inv(A)

x = np.linalg.solve(A,B)
# so x gives the solution x1 and x2, which are our scalers to multiply our coloumn vectors
scaled_A1 = x[0]*A[0]
scaled_A2 = x[1]*A[1]


plt.plot((0,A[0][0]),(0,A[1][0]),label = "First column vector A1")
plt.plot((0,A[0][1]),(0,A[1][1]), label = "Second column vector A2")
plt.plot((0,scaled_A1[0]),(0,scaled_A1[1]),label ="A1 scaled by x[0]")
plt.plot((scaled_A1[0],B[0]),(scaled_A1[1],B[1]), label = "A2 scaled by x[1]")
plt.plot((B[0],0),(B[1],0),label = 'Linear combination of A1 and A2')

plt.legend(loc='upper right', fontsize=10.5)


plt.show()