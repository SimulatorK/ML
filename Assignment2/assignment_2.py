
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
import os

from sklearn.metrics import accuracy_score,f1_score
import math

## Import mlrose
if __name__ == '__main__':
    os.chdir(os.path.split(__file__)[0])

os.chdir('../mlrose')
from mlrose.mlrose.fitness import (OneMax, FlipFlop, FourPeaks, SixPeaks, ContinuousPeaks,
                      Knapsack, TravellingSales, Queens, MaxKColor, 
                      CustomFitness)
from mlrose.mlrose.opt_probs import DiscreteOpt, ContinuousOpt, TSPOpt
from mlrose.mlrose.decay import GeomDecay, ArithDecay, ExpDecay, CustomSchedule
from mlrose.mlrose.algorithms import (hill_climb, random_hill_climb, simulated_annealing,
                         genetic_alg, mimic)
from mlrose.mlrose.neural import NeuralNetwork, LinearRegression, LogisticRegression
os.chdir('../Assignment2')

###############################################################################

# Load Data

fitness = Queens()

# Define optimization problem object
problem = DiscreteOpt(length = 8, fitness_fn = fitness, maximize=False, max_val=8)

# Define decay schedule
schedule = ExpDecay()

# Solve using simulated annealing - attempt 1         
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
best_state, best_fitness = simulated_annealing(problem, schedule = schedule, max_attempts = 10, 
                                                      max_iters = 1000, init_state = init_state,
                                                      random_state = 1)

print('The best state found is: ', best_state)

print('The fitness at the best state is: ', best_fitness)

# Solve using simulated annealing - attempt 2
best_state, best_fitness = simulated_annealing(problem, schedule = schedule, max_attempts = 100, 
                                                      max_iters = 1000, init_state = init_state,
                                                      random_state = 1)

print(best_state)

print(best_fitness) 

















