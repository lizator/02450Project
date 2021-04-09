import numpy as np
import pickle


# loading estimates

pickle_a = open("ClassificationData/a.pickle", "rb")
pickle_b = open("ClassificationData/b.pickle", "rb")
pickle_c = open("ClassificationData/c.pickle", "rb") 

y_est_a = pickle.load(pickle_a)  # K-Nearest
y_est_b = pickle.load(pickle_b)  # Logistic Regression
y_est_c = pickle.load(pickle_c)  # BaseLine

pickle_a.close()
pickle_b.close()
pickle_c.close()

# loading done
