import numpy as np
import xlrd

# load doc
doc = xlrd.open_workbook('./student-mat.xlsx').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=8)


# Setting up columns of data
sex = doc.col_values(1,1,396) # Converted: M == 0, F == 1
Age = doc.col_values(2,1,396) 
Medu = doc.col_values(3,1,396) 
Fedu = doc.col_values(4,1,396)
study = doc.col_values(5,1,396)
absence = doc.col_values(6,1,396)
g3 = doc.col_values(7,1,396)
maxEdu = []
for i in range(len(Medu)):
    maxEdu.append(max(Medu[i],Fedu[i]))


yrow = maxEdu   # what column is used for grouping
classNames = sorted(set(yrow))
classDict = dict(zip(classNames,range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.array([classDict[value] for value in yrow])

# Preallocate memory, then extract data to matrix X
X = np.empty((395,8))
for i in range(8):
    X[:,i] = np.array(doc.col_values(i,1,396)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)
