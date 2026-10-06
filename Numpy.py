#Creating an array
import numpy as np

x = np.array([[1,2,3,5],
              [2,3,5,8]])

print(x) # just print
print(x.shape) #rows,column
print(x.ndim) #number of axes x or y or z
print(x.size)#total number of elements rows*column

# Indexing
#           column
#          0   1   2
 #       ┌──────────
#row 0   │ 10  20  30
#row 1   │ 40  50  60
#row 2   │ 70  80  90

import numpy as np
A = np.array([
    [10,20,30],
    [40,50,60],
    [70,80,90]
])

print(A[0])
print(A[2,2])

#Slicing 1D
import numpy as np
x = np.array([10,20,30,40,50,60])
print(x[1:4])
print(x[2:]) # start at index 2 go till the end 30 40 50 60
print(x[:4]) # start from beginning and stop before index 4 10 20 30 40
# this means index 1 2 3 so 20 30 40

#slicing 2D

import numpy as np

A = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]

])

print(A[0:2, 1:3]) # row 0 and 1 is 0:2 and column 1 and 2 is 1:3
# so first take rows 0 and 1 then from there take columns 1 and 2 resulting in  2356

#Creating arrays automatically
import numpy as np
A = np.zeros((2,3))
print(A) #Create an array containing zeroes with 2 rows and 3 columns
print(A.shape)
B = np.ones((2,3)) # put ones instead of zeroes into 2 rows 3 columns

#Creates sequence of numbers

import numpy as np
x = np.arange(0,10)
print(x) #123456789 is the answer so its arange(start,stop,step)

#Element wise operations
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b) # 579
print(a - b) # 4 10 18
#A = [[1,2],
  #   [3,4]]

#B = [[5,6],
 #    [7,8]]
print(a * b) # 5 12 21 32 normal multiplication
print(a @ b) # matrix multiplication 19 22 43 50

#Broadcasting
import numpy as np

x = np.array([1, 2, 3])

print(x + 5)

#output is 6 7 8 bcz python took 5 as 5 5 5 + 1 2 3
# so broadcasting is basically expanding into comfortable shapes
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

b = np.array([10, 20, 30])

print(A + b)
#this will give smth likr 123456+ 102030 102030
#Vectorization
import numpy as np

x = np.array([1, 2, 3, 4, 5])

result = x * 10

print(result)

# this gives us 10 20 30 40 50 and we didnt need to use append function here just directly and no forming a new array

#More vectorization
# y = x**2 this means square every value

#DOT PRODUCT
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.dot(a, b)

print(result) # this gives u 4+10+18 = 32
# for 2 ID vectors a@b also gives the dot product
import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

C = A @ B

print(C) # the result is 1(5)+2(7) = 19 and so goes on
# Matrix shape rule
# MATRIX SHAPE RULE

# This is very important for ML.

# Suppose:

# A = (2 × 3)
# B = (3 × 4)

# Then:

# A @ B

# works because the inside numbers match:

# (2 × 3)
#       ↓
# (3 × 4)

# The result has the outside dimensions:

# (2 × 4)

# So:

# $$ (2\times3)(3\times4)=(2\times4) $$
# Memorize:
# ( A × B ) @ ( B × C )
#        ↓
#      ( A × C )

# The inside must match

# Matrix multiplied with vector
import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])

x = np.array([10, 20])

result = A @ x

print(result)

# so A is 2 cross 2 and x is 2, therefore A@x is 2, and thus 50 110

# Transpose
#switch rows and columns 
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A)
print(A.shape)

B = A.T

print(B)
print(B.shape)

# see how we wrote B = A.T this does the transpose
# and thus shape changes from 2,3 to 3,2

#Y = WX+B where w is the weight and b is the bias

import numpy as np

x = np.array([1, 2, 3, 4, 5])

y_actual = np.array([5, 7, 9, 11, 13])

w = 2
b = 3

y_pred = w * x + b

error = y_actual - y_pred

print("Prediction:", y_pred)
print("Error:", error)

# the actual values are the numerically solved ones and this is what the model predicted through the y predictions
# we then find the errors by doing actual-predicted

#Summarizing the ERRORS USING MSE

import numpy as np

x = np.array([1, 2, 3, 4, 5])

y_actual = np.array([5, 7, 9, 11, 13])

w = 2
b = 3

y_pred = w * x + b

error = y_actual - y_pred

mse = np.mean(error ** 2) #MEAN OF EVERY SQUARED ERROR this squaring makes negative numbers positive

print("Prediction:", y_pred)
print("Error:", error)
print("MSE:", mse)







