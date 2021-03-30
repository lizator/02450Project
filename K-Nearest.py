# exercise 6.3.2

from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

# loading data and removing G3
from loadData import *
from rmG3 import *

# Maximum number of neighbors
L=30

# Outer folds
K=10

# Inner folds
useLeaveOneOut = True
K2=10


# Script running

if useLeaveOneOut:
    CV = model_selection.LeaveOneOut()
else:
    CV = model_selection.KFold(n_splits=K2, shuffle=True)
CV_outer = model_selection.KFold(n_splits=K,shuffle=True)

folds = []

for par_index, verification_index in CV_outer.split(X, y):
    print("start of fold {0}".format(len(folds)+1))
    i=0
    errors = np.zeros((len(par_index),L))
    
    X_par = X[par_index,:]
    y_par = y[par_index]
    
    X_verification = X[verification_index,:]
    y_verification = y[verification_index]
    
    for train_index, test_index in CV.split(par_index):
        if useLeaveOneOut:
            mx = len(par_index)
        else:
            mx = K2
            
        print('Crossvalidation fold: {0}/{1}'.format(i+1,mx))    
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
    
        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train, y_train);
            y_est = knclassifier.predict(X_test);
            errors[i,l-1] = np.sum(y_est!=y_test)
    
        i+=1
        
    # Plot the classification error rate
    figure()
    plot(range(1,L+1),100*sum(errors,0)/N)
    xlabel('Number of neighbors, Fold {0}'.format(len(folds)+1))
    ylabel('Classification error rate (%)')
    show()
    par_best = np.argmin(sum(errors,0))+1
    knclassifier = KNeighborsClassifier(n_neighbors=par_best);
    knclassifier.fit(X_par, y_par);
    y_est = knclassifier.predict(X_verification);
    folds.append([par_best, 100*np.sum(y_est!=y_verification)/(len(y_verification))])
    print("")

for i in range(len(folds)):
    print("Fold {0}: \tbest={1}   \terror_rate={2}".format(i+1, folds[i][0], folds[i][1]))


                  

                  