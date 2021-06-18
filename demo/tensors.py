import torch
import numpy as np

data = [[1,2],[3,4]]
x_data= torch.tensor(data)
#print(x_data)
#print(x_data.device)

#define a matrix
mat1 = torch.ones(4,4)
mat2=torch.zeros(4,4)
mat1[:,1] = 0
mat1[:,2] =2
#print(mat1)

#Opertaions on matrix
matcat = torch.cat([mat1, mat2], dim=-0)
#print(matcat)

#mat * mat-transpose
multiply_mat = mat1 @ mat1.T
multiply_elementwise = mat1*mat1.T
print("matrix is ", mat1)
print(multiply_mat)
print(multiply_elementwise)
