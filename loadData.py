import numpy as np
import xlrd

# load doc
doc = xlrd.open_workbook('./student-mat.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=7)

# int to chosing what column is the y and the grouping of the dataset
col = 5

# Setting up columns of data
Female = doc.col_values(0,1,396) # Converted: M == 0, F == 1
Male = doc.col_values(1,1,396) 
Age = doc.col_values(2,1,396) 
study = doc.col_values(3,1,396)
absence = doc.col_values(4,1,396)
g3 = doc.col_values(5,1,396)
maxEdu = doc.col_values(6,1,396)

index = 0
while index != len(Female): #Removing all instances where g3 is 0
    if g3[index] < 0.5:
        Female = Female[:index] + Female[index+1:]
        Male = Male[:index] + Male[index+1:]
        Age = Age[:index] + Age[index+1:]
        study = study[:index] + study[index+1:]
        absence = absence[:index] + absence[index+1:]
        g3 = g3[:index] + g3[index+1:]
        maxEdu = maxEdu[:index] + maxEdu[index+1:]
    else:
        index += 1

lg3=[]
for index in g3:
    if index <10:
        lg3.append(0)
    else:
        lg3.append(1)

collected = [Female, Male, Age, study, absence, g3, maxEdu]
yrow = collected[col]   # what column is used for grouping
classNames = sorted(set(yrow))
classDict = dict(zip(classNames,range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.array([classDict[value] for value in yrow])

# Preallocate memory, then extract data to matrix X
X = np.empty((395,7))
for i in range(7):
    X[:,i] = np.array(doc.col_values(i,1,396)).T


index = 0
while index < len(X):
    if X[index][5] < 0.5:
        X = np.vstack((X[:index], X[index+1:]))
    else:
        index += 1

"""for row in X:
    if row[col] <7:
        row[col] = 0
    elif row[col] < 12:
        row[col] = 1
    elif row[col] < 18:
        row[col] = 2
    else:
        row[col] = 3"""
        
# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)
