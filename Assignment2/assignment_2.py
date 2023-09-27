
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

from sklearn import preprocessing
from  sklearn import model_selection
## Import mlrose
if __name__ == '__main__':
    os.chdir(os.path.split(__file__)[0])

os.chdir('../mlrose')
# =============================================================================
# from mlrose.fitness import (OneMax, FlipFlop, FourPeaks, SixPeaks, ContinuousPeaks,
#                       Knapsack, TravellingSales, Queens, MaxKColor, 
#                       CustomFitness)
# from mlrose.opt_probs import DiscreteOpt, ContinuousOpt, TSPOpt
# from mlrose.decay import GeomDecay, ArithDecay, ExpDecay, CustomSchedule
# from mlrose.algorithms import (hill_climb, random_hill_climb, simulated_annealing,
#                          genetic_alg, mimic)
# from mlrose.neural import NeuralNetwork, LinearRegression, LogisticRegression
# 
# =============================================================================
os.chdir('../mlrose-hiive')
# Import mlrose-hiive modules
from mlrose_hiive.algorithms.ga import (genetic_alg)
from mlrose_hiive.algorithms.sa import (simulated_annealing)
from mlrose_hiive.algorithms.hc import (hill_climb)
from mlrose_hiive.algorithms.rhc import (random_hill_climb)
from mlrose_hiive.algorithms.gd import (gradient_descent)
from mlrose_hiive.algorithms.mimic import (mimic)
from mlrose_hiive.algorithms.decay import GeomDecay, ArithDecay, ExpDecay, CustomSchedule
from mlrose_hiive.algorithms.crossovers import OnePointCrossOver, UniformCrossOver, TSPCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator, DiscreteMutator, SwapMutator, ShiftOneMutator
from mlrose_hiive.fitness import (OneMax, FlipFlop, FourPeaks, SixPeaks, ContinuousPeaks,
                      Knapsack, TravellingSales, Queens, MaxKColor,
                      CustomFitness)
from mlrose_hiive.neural import NeuralNetwork, LinearRegression, LogisticRegression, _nn_core, NNClassifier
from mlrose_hiive.neural.activation import (identity, relu,leaky_relu, sigmoid, softmax, tanh)
from mlrose_hiive.neural.fitness import NetworkWeights
from mlrose_hiive.neural.utils.weights import (flatten_weights, unflatten_weights)

from mlrose_hiive.gridsearch import GridSearchMixin

from mlrose_hiive.opt_probs import DiscreteOpt, ContinuousOpt, KnapsackOpt, TSPOpt, QueensOpt, FlipFlopOpt, MaxKColorOpt

from mlrose_hiive.runners import GARunner, MIMICRunner, RHCRunner, SARunner, NNGSRunner, SKMLPRunner
from mlrose_hiive.runners import (build_data_filename)
from mlrose_hiive.generators import (MaxKColorGenerator, QueensGenerator, FlipFlopGenerator, TSPGenerator, KnapsackGenerator,
                         ContinuousPeaksGenerator)

from mlrose_hiive.samples import SyntheticData
from mlrose_hiive.samples import (plot_synthetic_dataset)
os.chdir('../Assignment2')

### Import exponential algorithm for TSP problem
import held_karp as hk

# Master Inputs
###############################################################################
seed = 903860493
verbose = True
max_iters = 2000
max_attempts = 20
# MIMIC pop_size
pop_size = 500
restarts = 0
mutation_prob = 0.01

n_jobs = 20

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
    vPrint(f'RHC:\n\tBest Fit: {best_fitness_rhc}\n\tSolve Time: {solvetime_rh}')
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
    vPrint(f'SA:\n\tBest Fit: {best_fitness_sa}\n\tSolve Time: {solvetime_sa}')
    plt.plot(fitness_curve_sa,'r-',label='SA')
    
    # Genetic Algorithm
    start = time.time()
    best_state_ga, best_fitness_ga, fitness_curve_ga = genetic_alg(problem = problem,
                                                                   pop_size = pop_size,
                                                          max_iters = max_iters,
                                                          max_attempts = int(max_iters/2), # GA seems limited by attempts not iters
                                                          mutation_prob = mutation_prob,
                                                          curve = True,
                                                          random_state = seed,
                                                          )
    end = time.time()
    solvetime_ga = round(end - start,3)
    vPrint(f'GA:\n\tBest Fit: {best_fitness_ga}\n\tSolve Time: {solvetime_ga}')
    plt.plot(fitness_curve_ga,'g-',label='GA')
    
    # MIMIC
    start = time.time()
    best_state_m, best_fitness_m, fitness_curve_m = mimic(problem = problem,
                                                    pop_size = pop_size,
                                                    max_iters = max_iters,
                                                    max_attempts = max_attempts,
                                                    curve = True,
                                                    random_state = seed,
                                                    )
    end = time.time()
    solvetime_m = round(end - start,3)
    vPrint(f'MIMIC:\n\tBest Fit: {best_fitness_m}\n\tSolve Time: {solvetime_m}')
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
    plt.title(title)
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

n_cities = [100,]
for n in n_cities:
        
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
    
    title = f'TravellingSales_{n}'
    runProblem(problem=problem,title=title,**kwargs)
# =============================================================================
# 
# ###############################################################################
# # One Peak Problem
# ###############################################################################
# print('######################################################################')
# print('One Peak Problem')
# print('######################################################################')
# 
# bit_sizes = [10,100,1000,5000]
# for bits in bit_sizes:
#     bits = 1000
#     fitness = OneMax() 
#     problem = DiscreteOpt(length = bits, fitness_fn = fitness,
#                             maximize = True, max_val = 2)
#     
#     ##################################
#     # Solve using each algorithm
#     ##################################
#     kwargs = {
#             'restarts':restarts,
#             'max_iters':max_iters,
#             'max_attempts':max_attempts,
#             'pop_size':pop_size,
#             'mutation_prob':mutation_prob,
#               }
#     title = f'OnePeak_{bits}bits'    
#     runProblem(problem=problem,title=title,**kwargs)
# 
# =============================================================================
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
# =============================================================================
# 
# ###############################################################################
# # Knapsack Problem
# ###############################################################################
# print('######################################################################')
# print('Knapsack Problem')
# print('######################################################################')
# n_items = 50
# weights = [0] # First item has zero weight and zero value
# values = [0]
# weights.extend([random.randint(1,n_items) for _ in range(n_items - 1)])
# values.extend([random.randint(0,n_items) for _ in range(n_items - 1)])
# max_weight_pct = 1
# weights = np.array(weights)
# values = np.array(values)
# 
# # Custom Knapsack Fitness Function
# def custom_knapsack(state, weights, values, max_weight_pct):
#     """
#         State is the items to use in the knapsack.
#         Weights is the corresponding weight and values is the corresponding value
#         
#         Use with DiscreteOpt, number of states should match length and be integers
#     """
#     _w = np.ceil(max_weight_pct * np.sum(np.array(weights)))
#     #vPrint(f'Max weight: {_w}')
#     tot_weight = 0
#     tot_value = 0
#     for s in state:
#         s = int(s)
#         tot_weight += weights[s]
#         tot_value += values[s]
#     #vPrint(f'Total weight: {tot_weight}')
#     if tot_weight <= _w:
#         return tot_value
#     else:
#         return 0
# 
# kwargs = {'weights':weights,
#           'values':values,
#           'max_weight_pct':max_weight_pct,
#           }
# 
# fitness = CustomFitness(fitness_fn = custom_knapsack, problem_type='discrete', **kwargs)
#     
# # =============================================================================
# # fitness = Knapsack(weights = weights, values = values, max_weight_pct = max_weight_pct)
# # =============================================================================
# 
# problem = DiscreteOpt(length = n_items, fitness_fn = fitness, maximize=True, max_val=n_items)
# 
# ##################################
# # Solve using each algorithm
# ##################################
# kwargs = {
#         'restarts':restarts,
#         'max_iters':max_iters,
#         'max_attempts':max_attempts,
#         'pop_size':pop_size,
#         'mutation_prob':mutation_prob,
#           }
# 
# runProblem(problem=problem,title='Knapsack',**kwargs)
# 
# =============================================================================

###############################################################################
###############################################################################
# Neural Network Optimization
###############################################################################
###############################################################################


## Select Data Sets
mushroomClassificationFile = r"data/secondary+mushroom+dataset/MushroomDataset/secondary_data.csv"
ds = pd.read_csv(mushroomClassificationFile)

#Train Test Split for all experiments 
test_size = 0.2

# Split data

## Get X, Y data for test and train
Xdata = ds.iloc[:,1:-1]
Ydata = ds.iloc[:,-1]

## PRE-PROCESS ALL DATA
for col in range(Xdata.shape[1]):
    le = preprocessing.LabelEncoder()
    Xdata.iloc[:,col] = le.fit_transform(Xdata.iloc[:,col])
    Ydata = le.fit_transform(Ydata)
    
Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(Xdata,
                                                   Ydata,
                                                   test_size=test_size,
                                                   random_state=seed)

scaler = preprocessing.MinMaxScaler()

Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

############################
# Neural Network Model - run each optimization algorithm
############################

algos = ['random_hill_climb', 'simulated_annealing','genetic_alg',]
actives = ['identity', 'relu', 'sigmoid','tanh']
max_iters = 100
early_stopping = True
mutation_probs = np.linspace(0.001,0.1,10)
h_nodes = [[int(x)] for x in np.linspace(5, 25,5)]
pop_sizes = list(map(int,np.linspace(50,500,5)))

fig, ax = plt.subplots(figsize = (12,10),dpi = 200)
i = 0
ii = 0
best_model = None
best_f1 = 0
for algo in algos:

    model = NeuralNetwork(algorithm = algo,
                          activation = 'relu',
                          hidden_nodes = [Xdata.shape[1]],
                          mutation_prob = 0.01,
                          max_iters = max_iters,
                          curve = True, 
                          random_state = seed)
    
    start = time.time()
    model.fit(Xtrain,Ytrain)
    end = time.time()
    
    fit_time = end - start
    
    score = model.score(Xtest,Ytest)
    
    y_pred = model.predict(Xtest)
    
    f1 = f1_score(Ytest,y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_model = model

    vPrint(f'Algorithm: {algo}\n\tScore = {score}\n\tF1 = {f1}\n\tFit Time = {fit_time}')
    
    ## Add data to plot
    ax.plot(model.fitness_curve[:,0],label=f'{algo}')

plt.grid()
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('Images/NN_optimization.png')
plt.show()



### Optimal Algorithm Search
best_model_alg = best_model.algorithm

param_grid_nn = {
    'activation':actives,
    'algorithm':[best_model_alg,],
    'mutation_prob':mutation_probs,
    'hidden_nodes':h_nodes,
    }

model = NeuralNetwork(curve = True,
                          random_state = seed)

for act in actives:
    for algo in algos:
        if algo == 'genetic_alg':
            for mutation_prob in mutation_probs:
                for hn in h_nodes:
                    for pop_size in pop_sizes:         
                        model = NeuralNetwork(algorithm = algo,
                                              activation = act,
                                              hidden_nodes = hn,
                                              mutation_prob = mutation_prob,
                                              pop_size = pop_size,
                                              max_iters = max_iters,
                                              curve = True, 
                                              random_state = seed)

        else:
            for hn in h_nodes:
                            
                model = NeuralNetwork(algorithm = algo,
                                      activation = act,
                                      hidden_nodes = hn,
                                      mutation_prob = mutation_prob,
                                      max_iters = max_iters,
                                      curve = True, 
                                      random_state = seed)












