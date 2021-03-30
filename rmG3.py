# This code assumes that loadData already has been imported

# this wcript will remove G3 from X and attributeNames

import numpy as np
from loadData import *

X = np.hstack((X[:,:5], X[:,6:]))

attributeNames.pop(5)