
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For simulated annealing
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
#from scipy.optimize import basinhopping

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.datasets import make_classification

from sklearn import cross_validation
from sklearn.metrics import accuracy_score,f1_score
import math


## Blood caffeine concetration 
# https://matheducators.stackexchange.com/questions/1550/optimization-problems-that-todays-students-might-actually-encounter

def caffeine(t, alpha1, beta1, dose1,
             delay, alpha2, beta2, dose2):
    
    d1 = (dose1 / (1 - beta1/alpha1) ) * (np.exp(-beta1 * t) - np.exp(-alpha1 * t) ) 

    t2 = t - delay
    
    if t2 > 0:
        d2 = (dose2 / (1 - beta2/alpha2) ) * (np.exp(-beta2 * t2) - np.exp(-alpha2 * t2) ) 
    else:
        d2 = 0

    return d1 + d2




















