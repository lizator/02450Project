import torch
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net

#Load data
from loadData import *

#Extract G3 from X (the variable we want to predict)
y = X[:,5]

#Remove G3 from X
X = np.delete(X, 5, 1)
M = M-1

# Create crossvalidation partition for evaluation
K1 = 5
K2 = 5
CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
Error_test_baseline = np.empty((K1, 1))
Error_test_linear_regression = np.empty((K1, 1))
Error_test_ANN = np.empty((K1, 1))
mu = np.empty((K1, M))
sigma = np.empty((K1, M))
w_rlr = np.empty((M+1, K1))
opt_lambdas_rlr = np.empty((K1, 1))

n_replicates = 1 # number of networks trained in each k-fold
n_hidden_units = [1, 2, 5] # number of hidden units in the single hidden layer
max_iter = 10000 # Train for a maximum of 10000 steps, or until convergence (see help for the
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

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
    Error_test_baseline[k] = np.square(y_test-baselinePrediction).sum(axis=0)/y_test.shape[0]


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
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    # Compute mean squared error with regularization with optimal lambda
    Error_test_linear_regression[k] = np.square(y_test - X_test_reg @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]


    ### ARTIFICIAL NEURAL NETWORK ###

    errors = np.zeros((len(n_hidden_units), K2))  # make a list for storing generalizaition error for each model

    # Inner loop (K2)
    for (k, (train_index_ann, test_index_ann)) in enumerate(CV_inner.split(X_train, y_train)):

        X_train_ann = torch.Tensor(X_train[train_index_ann, :])
        y_train_ann = torch.Tensor(y_train[train_index_ann])
        X_test_ann = torch.Tensor(X_train[test_index_ann, :])
        y_test_ann = torch.Tensor(y_train[test_index_ann])

        # Model loop
        j = 0
        for i in n_hidden_units:

            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, i),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function, //todo kan ikke fjerne dette .. ?
                torch.nn.Linear(i, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_ann,
                                                               y=y_train_ann,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)

            # Determine estimated class labels for test set
            y_test_est_ann = net(X_test_ann)

            # Determine errors and errors
            se = (y_test_est_ann.float() - y_test_ann.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test_ann)).data.numpy().mean()  # mean
            errors[j][k] = mse

            j += 1








    # 10-fold cross validate ANN to find optimal parameters (inner loop)
    # Test optimal model on X/Y_test
    # Save result

    k += 1

for i in range(K1):
    print(Error_test_baseline[i])


print("\nLinReg")

for i in range(K1):
    print(Error_test_linear_regression[i])


# E_{gen} som gennemsnittet over alle K1 loops
# Print results

