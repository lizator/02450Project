# used from Exercise 4.2.5

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show)

import numpy as np
import copy
from toolbox_02450.similarity import similarity

# requires data from dataset
from loadData import *

from noiseCreator import NoiseCreator

Xcopy = copy.deepcopy(X)
for index in range(len(Xcopy)):
    Xcopy[index] = NoiseCreator.createNoise(Xcopy[index], 0.2)

figure(figsize=(24,20))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            collected_mask = []
            for boo in class_mask:
                collected_mask.append(True)
            plot(np.array(Xcopy[class_mask,m2]), np.array(Xcopy[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2] + ", Correlation: %.4f " % np.corrcoef(np.array(X[collected_mask,m2]), np.array(X[collected_mask,m1]))[0,1])
            else:
                xticks([])
                xlabel("Correlation: %.4f" % np.corrcoef(np.array(X[collected_mask,m2]), np.array(X[collected_mask,m1]))[0,1])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)

show()

for c in range(C):
    class_mask = (y==c)
    collected_mask = []
    for boo in class_mask:
        collected_mask.append(True)
    plot(np.array(Xcopy[class_mask,6]), np.array(Xcopy[class_mask,5]), '.')

ylabel("G3")
xlabel("maxEdu, Correlation: %.4f" % np.corrcoef(np.array(X[collected_mask,6]), np.array(X[collected_mask,5]))[0,1])
legend(classNames)
show()
