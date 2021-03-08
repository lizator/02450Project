import matplotlib.pyplot as plt
from scipy.stats import zscore

from loadData import *

import random

def createNoise(array, noise):
    noisyarray = []
    for index in array:
        noisyarray.append(index + random.uniform(-noise, noise))

    return noisyarray

# Sex
plt.hist(X[:,0:1], bins=np.arange(0,3)-0.5, edgecolor='black')
plt.xticks([0, 1], ["Male", "Female"])
plt.ylabel("Frequency")
# plt.savefig('ageHist.png')
plt.show()

# Age
plt.hist(X[:,2], bins=np.arange(15,24)-0.5, edgecolor='black')
plt.xticks(range(15,23))
plt.ylabel("Frequency")
plt.xlabel("Age (yrs)")
#plt.savefig('ageHist.png')
plt.show()

# Studytime
plt.hist(X[:,3], bins=np.arange(1,6)-0.5, edgecolor='black')
plt.xticks(range(1,5))
plt.ylabel("Frequency")
plt.xlabel("Studytime (hrs/week)")
#plt.savefig('studytimeHist.png')
plt.show()

# Absence
plt.hist(X[:,4]/93*100, bins=np.arange(0,90,5), edgecolor='black')
plt.xticks(range(0,90,5))
plt.ylabel("Frequency")
plt.xlabel("Absence (%)")
#plt.savefig('absenceHist.png')
plt.show()

# Grade
plt.hist(X[:,5], bins=np.arange(0,21,2), edgecolor='black')
plt.xticks(range(0,21,2))
plt.ylabel("Frequency")
plt.xlabel("Grade")
#plt.savefig('gradeHist.png')
plt.show()

# Max(medu, fedu)
plt.hist(X[:,6], bins=np.arange(1,6)-0.5, edgecolor='black')
plt.xticks(range(1,5))
plt.xlabel("Max(medu, fedu)")
plt.ylabel("Frequency")
#plt.savefig('eduHist.png')
plt.show()

# Scatterplot
plt.figure(figsize=(5,10))
plt.scatter(createNoise(np.ones(len(X[:,0:1])), 0.35), createNoise(X[:,0:1], 0.4))
plt.scatter(createNoise(np.ones(len(X[:,2]))*2, 0.35), createNoise(X[:,2], 0.4))
plt.scatter(createNoise(np.ones(len(X[:,3]))*3, 0.35), createNoise(X[:,3], 0.4))
plt.scatter(createNoise(np.ones(len(X[:,4]))*4, 0.35), createNoise(X[:,4]/93*100, 0.4))
plt.scatter(createNoise(np.ones(len(X[:,5]))*5, 0.35), createNoise(X[:,5], 0.4))
plt.scatter(createNoise(np.ones(len(X[:,6]))*6, 0.35), createNoise(X[:,6], 0.4))
plt.xticks([1, 2, 3, 4, 5, 6], ['Sex', 'Age (yrs)', 'Studytime', 'Absence (%)', 'Grade', 'Max(fedu, medu)'], rotation=45)
# plt.savefig('scatter.png')
plt.show()

# Boxplot
boxX = np.hstack((X[:,2], X[:,4]/93*100, X[:,5])).reshape(-1,len(X[:,6]))
boxX = boxX.T
plt.boxplot(boxX)
plt.xticks([1, 2, 3], ['Age (yrs)', 'Absence (%)', 'Grade'])
# plt.savefig('box.png')
plt.show()

# Summary
for i in range(8):
    print("att: {}".format(attributeNames[i]))
    print("n = {}".format(X[:, i].size))
    print("mean = {}".format(X[:, i].mean()))
    print("s^2 = {}".format(X[:, i].var()))
    print("s = {}".format(X[:, i].std()))
    print("Q1 = {}".format(np.quantile(X[:, i], 0.25)))
    print("Q2 = {}".format(np.quantile(X[:, i], 0.50)))
    print("Q3 = {}".format(np.quantile(X[:, i], 0.75)))
    print()



