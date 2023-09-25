
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
import random
from sklearn.metrics import accuracy_score,f1_score
import math
import networkx as nx 

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
seed = 903860493
verbose = True

def vPrint(text: str = '', verbose: bool = verbose):
    if verbose:
        print(text)

###############################################################################
# Travelling Sales Optimization Problem
###############################################################################
max_attempts = 100
n = 15 # Number of cities
probConnect = 0.5
useDistances = True

# Create a connected graph to represent cities and routes

g = nx.erdos_renyi_graph(n, 0.5, seed=seed, directed=False)
coords = list(g.edges)

# Draw the city graph
labels = {g.nodes[i]: f'City {i}' for i in range(n)}

fig = plt.subplots(figsize = (12,8), dpi = 200)
nx.draw(g,labels = labels, with_labels=True)
plt.savefig('TravellingSales_CityGraph.png')
plt.show()

fitness = TravellingSales(coords = coords)

problem = TSPOpt(length = len(coords), fitness_fn = fitness)

# Define decay schedule
schedule = ExpDecay()

###############################################################################
# Solve using each algorithm
###############################################################################

# Random hill Climbing
fig = plt.subplots(figsize = (12,8), dpi = 200)

best_state, best_fitness, fitness_curve = random_hill_climb(problem = problem,
                                                            max_attempts = max_attempts,
                                                            curve = True,
                                                            random_state = seed,
                                                            )
plt.plot(fitness_curve,label='RHC')

# Simulated Annealing
best_state, best_fitness, fitness_curve = simulated_annealing(problem = problem,
                                                            max_attempts = max_attempts,
                                                            curve = True,
                                                            random_state = seed,
                                                            )
plt.plot(fitness_curve,label='SA')

# Genetic Algorithm
best_state, best_fitness, fitness_curve = genetic_alg(problem = problem,
                                                      max_attempts = max_attempts,
                                                      curve = True,
                                                      random_state = seed,
                                                      )
plt.plot(fitness_curve,label='GA')

# MIMIC
best_state, best_fitness, fitness_curve = mimic(problem = problem,
                                                pop_size = 1000,
                                                max_attempts = max_attempts,
                                                curve = True,
                                                random_state = seed,
                                                )
plt.plot(fitness_curve,label='MIMIC')


# Format Chart
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.grid()
plt.legend()
plt.savefig('TravellingSales_OptProblem_Comparision.png')
plt.show()













