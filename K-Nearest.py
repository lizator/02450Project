# exercise 6.3.2

from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot

# loading data and removing G3
#from loadData import *
from rmG3 import *
"""
mat_data = loadmat('../../02450Toolbox_Python/Data/wine2.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)
"""
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
    
    # setting baseline
    
    most = round(np.sum(y[par_index])/len(par_index))
    base_error_rate = np.sum(y[verification_index]!=most)/len(verification_index)*100
    
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
        
    # Doing logistical regression part
        
    # Standardize the training and set set based on training set mean and std
    mu = np.mean(X_verification, 0)
    sigma = np.std(X_verification, 0)
    
    X_train = (X_verification - mu) / sigma
    X_test = (X_par - mu) / sigma
    
    # Fit regularized logistic regression model 
    lambda_interval = np.logspace(-8, 2, 50)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    for k in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
        
        mdl.fit(X_train, y_verification)
    
        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
        
        train_error_rate[k] = np.sum(y_train_est != y_verification) / len(y_verification)
        test_error_rate[k] = np.sum(y_test_est != y_par) / len(y_par)
    
        w_est = mdl.coef_[0] 
        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    
    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]  
    
    
    
      
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
    folds.append([par_best, 100*np.sum(y_est!=y_verification)/(len(y_verification)), opt_lambda, min_error*100, base_error_rate])
    print("")
    
    plt.figure(figsize=(8,8))
    #plt.plot(np.log10(lambda_interval), train_error_rate*100)
    #plt.plot(np.log10(lambda_interval), test_error_rate*100)
    #plt.plot(np.log10(opt_lambda), min_error*100, 'o')
    plt.semilogx(lambda_interval, train_error_rate*100)
    plt.semilogx(lambda_interval, test_error_rate*100)
    plt.semilogx(opt_lambda, min_error*100, 'o')
    plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.ylabel('Error rate (%)')
    plt.title('Classification error')
    plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
    plt.ylim([0, 50])
    plt.grid()
    plt.show()  
    

for i in range(len(folds)):
    print("Fold {0}: \tK-nearest: best={1}\terror_rate={2}\tLogistical: best={3}\terror_rate={4}\tBase error_rate={5}".format(i+1, folds[i][0], format(folds[i][1], '.2f'), format(folds[i][2], '.8f'), format(folds[i][3], '.2f'), format(folds[i][4], '.2f')))

neigbour_sum = 0
logist_sum = 0
base_sum = 0
for l in folds:
    neigbour_sum += l[1]
    logist_sum += l[3]
    base_sum += l[4]
    
print("Neighbor estimate: {0}, logist estimate: {1}, Base estimate: {2}".format(neigbour_sum/K, logist_sum/K, base_sum/K))
                  

                  