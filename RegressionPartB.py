import torch
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net
from tabulate import tabulate
from numpy import savetxt
import time

start_time = time.time()

#Load data
from loadData import *

#Extract G3 from X (the variable we want to predict)
y = X[:,5]

#Remove G3 from X
X = np.delete(X, 5, 1)
M = M-1

# Create crossvalidation partition for evaluation
K1 = 10
K2 = 10
CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
Error_test_baseline_avg = np.empty((K1, 1))
Error_test_baseline = np.zeros(X.shape[0])
Error_test_linear_regression_avg = np.empty((K1, 1))
Error_test_linear_regression = np.zeros(X.shape[0])
Error_test_ANN_avg = np.empty((K1, 1))
Error_test_ANN = np.zeros(X.shape[0])
mu = np.empty((K1, M))
sigma = np.empty((K1, M))
w_rlr = np.empty((M+1, K1))
opt_lambdas_rlr = np.empty((K1, 1))

n_replicates = 3 # number of networks trained in each k-fold
n_hidden_units = [1,3,5,10,20] # number of hidden units in the single hidden layer
max_iter = 10000 # Train for a maximum of 10000 steps, or until convergence (see help for the
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
opt_hidden_units = np.empty((K1, 1))

k = 0
for train_index, test_index in CV_outer.split(X):

    # extract training and test set for current CV fold (outer fold)
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]


    ### BASELINE ###
    # Baseline does not need any optimization and can be evaluated directed on X/Y_test
    baselinePrediction = y_train.mean()
    Error_test_baseline_avg[k] = np.square(y_test - baselinePrediction).sum(axis=0) / y_test.shape[0]
    Error_test_baseline[test_index] = np.square(y_test - baselinePrediction)


    ### LINEAR REGRESSION ###
    # Add offset attribute
    X_train_reg = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
    X_test_reg = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)

    # 10-fold cross validate linear regression to find optimal lambda (inner loop)
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train_reg,
                                                                                                      y_train,
                                                                                                      lambdas,
                                                                                                      K2)
    opt_lambdas_rlr[k] = opt_lambda

    # Standardize data based on training set
    # Notice how we leave the offset attribute out
    mu[k, :] = np.mean(X_train_reg[:, 1:], 0)
    sigma[k, :] = np.std(X_train_reg[:, 1:], 0)

    X_train_reg[:, 1:] = (X_train_reg[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test_reg[:, 1:] = (X_test_reg[:, 1:] - mu[k, :]) / sigma[k, :]

    # Calculate weights for the optimal value of lambda, on entire training set
    Xty = X_train_reg.T @ y_train
    XtX = X_train_reg.T @ X_train_reg

    lambdaI = opt_lambda * np.eye(M+1)
    lambdaI[0, 0] = 0  # Do not regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    # Compute mean squared error with regularization with optimal lambda
    Error_test_linear_regression_avg[k] = np.square(y_test - X_test_reg @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    Error_test_linear_regression[test_index] = np.square(y_test - X_test_reg @ w_rlr[:, k])


    ### ARTIFICIAL NEURAL NETWORK ###

    errors = np.zeros((len(n_hidden_units), K2))  # make a list for storing generalizaition error for each model

    # Inner loop (K2)
    i = 0
    for train_index_ann, test_index_ann in CV_inner.split(X_train):

        X_train_ann = torch.Tensor(X_train[train_index_ann, :])
        y_train_ann = torch.Tensor(y_train[train_index_ann])
        X_test_ann = torch.Tensor(X_train[test_index_ann, :])
        y_test_ann = torch.Tensor(y_train[test_index_ann])

        # Model loop
        j = 0
        for n in n_hidden_units:

            inner_model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, n),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function, //todo kan ikke fjerne dette .. ?
                torch.nn.Linear(n, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(inner_model,
                                                               loss_fn,
                                                               X=X_train_ann,
                                                               y=y_train_ann,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)

            # Determine estimated class labels for test set
            y_test_est_ann = net(X_test_ann)

            # Determine errors
            se = (y_test_est_ann.float() - y_test_ann.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test_ann)).data.numpy().mean()  # mean
            errors[j][i] = mse

            j += 1
        i += 1

    # Save the best beforming number of hidden units from the inner loop
    opt_hidden_units[k] = n_hidden_units[np.argmin(np.mean(errors, axis=1))]
    opt_hidden_unit = opt_hidden_units[k][0].astype(int)

    # Compute error for best performing model for the outer loop
    outer_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, opt_hidden_unit),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function, //todo kan ikke fjerne dette .. ?
        torch.nn.Linear(opt_hidden_unit, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(outer_model,
                                                       loss_fn,
                                                       X=torch.Tensor(X_train),
                                                       y=torch.Tensor(y_train),
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)

    # Determine estimated class labels for test set
    y_test_est = net(torch.Tensor(X_test))

    # Determine errors
    se = (y_test_est.float() - torch.Tensor(y_test).float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(torch.Tensor(y_test))).data.numpy().mean()  # mean
    Error_test_ANN_avg[k] = mse
    Error_test_ANN[test_index] = np.square(y_test_est.float().data.numpy()[:,0] - torch.Tensor(y_test).float().data.numpy())

    k += 1

# Output data in required format
output_data = np.hstack((
    np.arange(K1).reshape(K1,1) + 1,
    opt_hidden_units,
    Error_test_ANN_avg,
    opt_lambdas_rlr,
    Error_test_linear_regression_avg,
    Error_test_baseline_avg
))

print(tabulate(
    output_data,
    headers=['i','ann_h','ann_err','lr_lambda','lr_err','base_err']))

print("\n--- %s seconds ---" % (time.time() - start_time))

# Export for statistical comparison of models in R
savetxt('Error_ANN.csv', Error_test_ANN, delimiter=',')
savetxt('Error_baseline.csv', Error_test_baseline, delimiter=',')
savetxt('Error_linear_regression.csv', Error_test_linear_regression, delimiter=',')
