import sklearn.linear_model as lm
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
from loadData import *
from rmG3 import *

for i in range(len(y)):
    if y[i] < 10:
        y[i] = 0
    else:
        y[i] = 1

C=2

model = lm.LogisticRegression(C=1/10)
model = model.fit(X,y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_pass_prob = model.predict_proba(X)[:, 0] 

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_pass_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_pass_prob[class1_ids], '.r')
xlabel('Data object (student sample)'); ylabel('Predicted prob. of class Fail');
legend(['Fail', 'Pass'])
ylim(-0.01,1.5)

show()

print(model.coef_[0])
print(attributeNames)
print("Offset: {0:.4f}".format(model.intercept_[0]))
