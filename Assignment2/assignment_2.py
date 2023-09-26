
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
from numpy import random
from sklearn.metrics import accuracy_score,f1_score
import math
import networkx as nx 
import time

## Import mlrose
if __name__ == '__main__':
    os.chdir(os.path.split(__file__)[0])

os.chdir('../mlrose')
from mlrose.fitness import (OneMax, FlipFlop, FourPeaks, SixPeaks, ContinuousPeaks,
                      Knapsack, TravellingSales, Queens, MaxKColor, 
                      CustomFitness)
from mlrose.opt_probs import DiscreteOpt, ContinuousOpt, TSPOpt
from mlrose.decay import GeomDecay, ArithDecay, ExpDecay, CustomSchedule
from mlrose.algorithms import (hill_climb, random_hill_climb, simulated_annealing,
                         genetic_alg, mimic)
from mlrose.neural import NeuralNetwork, LinearRegression, LogisticRegression
os.chdir('../Assignment2')

### Import exponential algorithm for TSP problem
import held_karp as hk

# Master Inputs
###############################################################################
seed = 903860493
verbose = True
max_iters = 3000
max_attempts = 20
# MIMIC pop_size
pop_size = 500
restarts = 0
mutation_prob = 0.4

def vPrint(text: str = '', verbose: bool = verbose):
    if verbose:
        print(text)

#Set random seed
np.random.seed(seed)
### Master Problem Run
def runProblem(problem,title: str, **kwargs):
    restarts = kwargs['restarts']
    max_iters = kwargs['max_iters']
    max_attempts = kwargs['max_attempts']
    pop_size = kwargs['pop_size']
    mutation_prob = kwargs['mutation_prob']
    ##################################
    # Solve using each algorithm
    ##################################
    fig = plt.subplots(figsize = (12,8), dpi = 200)
    # Random hill Climbing
    start = time.time()
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = random_hill_climb(problem = problem,
                                                                            restarts = restarts,
                                                                            max_iters = max_iters,
                                                                            max_attempts = max_attempts,
                                                                            curve = True,
                                                                            random_state = seed,
                                                                            )
    end = time.time()
    solvetime_rh = round(end - start,3)
    vPrint(f'RHC:\n\tBest State: {best_state_rhc}\n\tBest Fit: {best_fitness_rhc}\n\tSolve Time: {solvetime_rh}')
    plt.plot(fitness_curve_rhc,'b-',label='RHC')
    # Simulated Annealing
    start = time.time()
    best_state_sa, best_fitness_sa, fitness_curve_sa = simulated_annealing(problem = problem,
                                                                           schedule = schedule,
                                                                max_iters = max_iters,
                                                                max_attempts = max_attempts,
                                                                curve = True,
                                                                random_state = seed,
                                                                )
    end = time.time()
    solvetime_sa = round(end - start,3)
    vPrint(f'SA:\n\tBest State: {best_state_sa}\n\tBest Fit: {best_fitness_sa}\n\tSolve Time: {solvetime_sa}')
    plt.plot(fitness_curve_sa,'r-',label='SA')
    
    # Genetic Algorithm
    start = time.time()
    best_state_ga, best_fitness_ga, fitness_curve_ga = genetic_alg(problem = problem,
                                                                   pop_size = pop_size,
                                                          max_iters = max_iters,
                                                          max_attempts = max_attempts,
                                                          mutation_prob = mutation_prob,
                                                          curve = True,
                                                          random_state = seed,
                                                          )
    end = time.time()
    solvetime_ga = round(end - start,3)
    vPrint(f'GA:\n\tBest State: {best_state_ga}\n\tBest Fit: {best_fitness_ga}\n\tSolve Time: {solvetime_ga}')
    plt.plot(fitness_curve_ga,'g-',label='GA')
    
    # MIMIC
    try:
        start = time.time()
        best_state_m, best_fitness_m, fitness_curve_m = mimic(problem = problem,
                                                        pop_size = pop_size,
                                                        max_iters = max_iters,
                                                        max_attempts = max_attempts,
                                                        curve = True,
                                                        random_state = seed,
                                                        fast_mimic=True
                                                        )
        end = time.time()
        solvetime_m = round(end - start,3)
        vPrint(f'MIMIC:\n\tBest State: {best_state_m}\n\tBest Fit: {best_fitness_m}\n\tSolve Time: {solvetime_m}')
        plt.plot(fitness_curve_m,'k-',label='MIMIC')
    except:
        vPrint('Fast MIMIC failed, trying slow...')
        start = time.time()
        best_state_m, best_fitness_m, fitness_curve_m = mimic(problem = problem,
                                                        pop_size = pop_size,
                                                        max_iters = max_iters,
                                                        max_attempts = max_attempts,
                                                        curve = True,
                                                        random_state = seed,
                                                        fast_mimic=False
                                                        )
        end = time.time()
        solvetime_m = round(end - start,3)
        vPrint(f'MIMIC:\n\tBest State: {best_state_m}\n\tBest Fit: {best_fitness_m}\n\tSolve Time: {solvetime_m}')
        plt.plot(fitness_curve_m,'k-',label='MIMIC')
        
    # Format Chart
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'Images/{title}_Comparision.png')
    plt.show()
    
    # Plot solve times
    fig = plt.subplots(figsize=(12,10),dpi = 200)
    plt.bar('RHC',solvetime_rh)
    plt.bar('SA',solvetime_sa)
    plt.bar('GA',solvetime_ga)
    plt.bar('MIMIC',solvetime_m)
    plt.ylabel('Solution Time (sec)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'Images/{title}_SolveTime.png')
    plt.show()
    
###############################################################################
# Travelling Sales Optimization Problem
###############################################################################

print('######################################################################')
print('Travelling Salesman Problem')
print('######################################################################')

n = 50 # Number of cities
probConnect = 0.5
useDistances = True

# Create a connected graph to represent cities and routes

g = nx.erdos_renyi_graph(n, probConnect, seed=seed, directed=False)
while not nx.is_connected(g):
    n += 1
    vPrint('Graph not connected, increasing n...')
    g = nx.erdos_renyi_graph(n, 0.5, seed=seed, directed=False)
vPrint(f'N set to {n}')

# Add weights
for (u, v) in g.edges():
    g.edges[u,v]['weight'] = np.ceil(random.random()*10)
    
distances = [(u,v,w['weight']) for (u,v,w) in g.edges(data=True)]


## Create distance matrix
# =============================================================================
# dist_matrix = np.ones((n,n))
# for i in range(n):
#     for ii in range(n):
#         edge = (i,ii)
#         if edge in distances:
#             dist_matrix[i][ii] = -1
#         else:
#             dist_matrix[i][ii] = 1
# 
# # Compute held_karp
# brute_force = hk.held_karp(dist_matrix)
# brute_path = []
# for n, target in enumerate(brute_force[1]):
#     source = brute_force[1][n-1]
#     if source not in brute_path:
#         brute_path.append(source)
#         
#     shortest_p = nx.shortest_path(g,source,target)
#     for n_ in shortest_p:
#         if n_ != source and n_ != target:
#             brute_path.append(n)
#             
# print(f'Brute force path: {brute_path}')
# =============================================================================

# Draw the city graph
# =============================================================================
# labels = {list(g.nodes)[i]: f'{i}' for i in range(n)}
# =============================================================================
fig = plt.subplots(figsize = (12,8), dpi = 200)
pos=nx.spring_layout(g)
nx.draw(g, with_labels=True)
labels = nx.get_edge_attributes(g,'weight')
# =============================================================================
# nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
# =============================================================================
plt.title(f'{n} City Connected Graph')
plt.savefig('Images/TravellingSales_CityGraph.png')
plt.show()

# Get pos from spring layout
coords = [(c[0],c[1]) for c in pos.values()]

fitness = TravellingSales(coords = coords, )#distances = distances)

problem = TSPOpt(length = n, fitness_fn = fitness)

# Define decay schedule
schedule = ExpDecay()

##################################
# Solve using each algorithm
##################################
kwargs = {
        'restarts':restarts,
        'max_iters':max_iters,
        'max_attempts':max_attempts,
        'pop_size':pop_size,
        'mutation_prob':mutation_prob,
          }

runProblem(problem=problem,title='TravellingSales',**kwargs)

###############################################################################
# One Peak Problem
###############################################################################
print('######################################################################')
print('One Peak Problem')
print('######################################################################')

bits = 1000
fitness = OneMax() 
problem = DiscreteOpt(length = bits, fitness_fn = fitness,
                        maximize = True, max_val = 2)

##################################
# Solve using each algorithm
##################################
kwargs = {
        'restarts':restarts,
        'max_iters':max_iters,
        'max_attempts':max_attempts,
        'pop_size':pop_size,
        'mutation_prob':mutation_prob,
          }

runProblem(problem=problem,title='OnePeak',**kwargs)

###############################################################################
# Random Bit Match Problem
###############################################################################
print('######################################################################')
print('Random Bit Match')
print('######################################################################')

bits = 1000
mask = np.random.randint(0,2,bits)

## Fitness function returns successively higher values for multiple adjacent matched bits
## Function takes a  bit string that is meant to be matched
## Non matching bits deduct a point
def random_bit_match(state, mask):
    
    total_value = 0
    m = 0
    for i, s in enumerate(state):
        if s == mask[i]:
            m += 1
            total_value += m
        else:
            m = 0
            total_value -= 1
            total_value = max(total_value,0)
    
    return total_value

kwargs = {'mask':mask,}

fitness = CustomFitness(fitness_fn = random_bit_match, problem_type='discrete', **kwargs)

problem = DiscreteOpt(length = bits, fitness_fn = fitness, maximize=True, max_val=2)

##################################
# Solve using each algorithm
##################################
kwargs = {
        'restarts':restarts,
        'max_iters':max_iters,
        'max_attempts':max_attempts,
        'pop_size':pop_size,
        'mutation_prob':mutation_prob,
          }

runProblem(problem=problem,title='RandomBitMatch',**kwargs)


###############################################################################
# Knapsack Problem
###############################################################################
print('######################################################################')
print('Four Peaks Problem')
print('######################################################################')
bits = 1000
fitness = FourPeaks()

problem = DiscreteOpt(length = bits, fitness_fn = fitness, maximize = True, max_val = 2)
##################################
# Solve using each algorithm
##################################
kwargs = {
        'restarts':restarts,
        'max_iters':max_iters,
        'max_attempts':max_attempts,
        'pop_size':pop_size,
        'mutation_prob':mutation_prob,
          }

runProblem(problem=problem,title='FourPeaks',**kwargs)

###############################################################################
# Knapsack Problem
###############################################################################
print('######################################################################')
print('Knapsack Problem')
print('######################################################################')
n_items = 20
weights = [0] # First item has zero weight and zero value
values = [0]
weights.extend([random.randint(1,n_items) for _ in range(n_items - 1)])
values.extend([random.randint(0,n_items) for _ in range(n_items - 1)])
max_weight_pct = 1
weights = np.array(weights)
values = np.array(values)

# Custom Knapsack Fitness Function
def custom_knapsack(state, weights, values, max_weight_pct):
    """
        State is the items to use in the knapsack.
        Weights is the corresponding weight and values is the corresponding value
        
        Use with DiscreteOpt, number of states should match length and be integers
    """
    _w = np.ceil(max_weight_pct * np.sum(np.array(weights)))
    #vPrint(f'Max weight: {_w}')
    tot_weight = 0
    tot_value = 0
    for s in state:
        s = int(s)
        tot_weight += weights[s]
        tot_value += values[s]
    #vPrint(f'Total weight: {tot_weight}')
    if tot_weight <= _w:
        return tot_value
    else:
        return 0

kwargs = {'weights':weights,
          'values':values,
          'max_weight_pct':max_weight_pct,
          }

fitness = CustomFitness(fitness_fn = custom_knapsack, problem_type='discrete', **kwargs)
    
# =============================================================================
# fitness = Knapsack(weights = weights, values = values, max_weight_pct = max_weight_pct)
# =============================================================================

problem = DiscreteOpt(length = n_items, fitness_fn = fitness, maximize=True, max_val=n_items)

##################################
# Solve using each algorithm
##################################
kwargs = {
        'restarts':restarts,
        'max_iters':max_iters,
        'max_attempts':max_attempts,
        'pop_size':pop_size,
        'mutation_prob':mutation_prob,
          }

runProblem(problem=problem,title='Knapsack',**kwargs)













