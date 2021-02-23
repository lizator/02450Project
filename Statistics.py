import matplotlib.pyplot as plt

from loadData import *

# Sex
plt.hist(X[:,0], bins=np.arange(0,3)-0.5, edgecolor='black')
plt.xticks(range(0,2))
plt.ylabel("Frequency")
plt.xlabel("Sex")
# plt.savefig('sexHist.png')
plt.show()

# Age
plt.hist(X[:,1], bins=np.arange(15,24)-0.5, edgecolor='black')
plt.xticks(range(15,23))
plt.ylabel("Frequency")
plt.xlabel("Age (yrs)")
# plt.savefig('ageHist.png')
plt.show()

plt.boxplot(X[:,1])
plt.ylabel("Age (yrs)")
# plt.savefig('ageBox.png')
plt.show()

# Studytime
plt.hist(X[:,4], bins=np.arange(1,6)-0.5, edgecolor='black')
plt.xticks(range(1,5))
plt.ylabel("Frequency")
plt.xlabel("Studytime (hrs/week)")
# plt.savefig('studytimeHist.png')
plt.show()

plt.boxplot(X[:,4])
plt.ylabel("Studytime (hrs/week)")
# plt.savefig('studytimeBox.png')
plt.show()

# Absence
plt.hist(X[:,5]/93*100, bins=np.arange(0,90,5), edgecolor='black')
plt.xticks(range(0,90,5))
plt.ylabel("Frequency")
plt.xlabel("Absence (%)")
# plt.savefig('absenceHist.png')
plt.show()

plt.boxplot(X[:,5])
plt.ylabel("Absence (%)")
# plt.savefig('absenceBox.png')
plt.show()

# Grade
plt.hist(X[:,6], bins=np.arange(0,21,2), edgecolor='black')
plt.xticks(range(0,21,2))
plt.ylabel("Frequency")
plt.xlabel("Grade")
# plt.savefig('gradeHist.png')
plt.show()

plt.boxplot(X[:,6])
plt.ylabel("Grade")
# plt.savefig('gradeBox.png')
plt.show()

# Max(medu, fedu)
plt.hist(X[:,7], bins=np.arange(1,6)-0.5, edgecolor='black')
plt.xticks(range(1,5))
plt.xlabel("Max(medu, fedu)")
plt.ylabel("Frequency")
# plt.savefig('eduHist.png')
plt.show()

plt.boxplot(X[:,7])
plt.ylabel("Max(medu, fedu)")
# plt.savefig('eduBox.png')
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




