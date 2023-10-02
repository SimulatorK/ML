
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
import sklearn.metrics as skm
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
main_start = time.time()
seed = 903860493
seeds = list(map(int,np.random.randint(1,4,3)))
verbose = True
max_iters = 1000
max_attempts = 20
# MIMIC pop_size
pop_size = 500
restarts = 0
mutation_prob = 0.01
keep_pct = 0.2

n_jobs = 20

def vPrint(text: str = '', verbose: bool = verbose):
    if verbose:
        print(text)

#Set random seed

### Master Problem Run
def runProblem(problem,title: str, **kwargs):
   
    np.random.seed(seed)
    f = -1 if problem.maximize < 0 else 1
    
    restarts = int(kwargs['restarts'])
    max_iters = int(kwargs['max_iters'])
    max_attempts = int(kwargs['max_attempts'])
    pop_size = int(kwargs['pop_size'])
    mutation_prob = kwargs['mutation_prob']
    keep_pct = kwargs['keep_pct']
    ##################################
    # Solve using each algorithm
    ##################################
    fig = plt.subplots(figsize = (12,8), dpi = 200)
    # Random hill Climbing
    start = time.time()
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = random_hill_climb(problem = problem,
                                                                            restarts = restarts,
                                                                            max_iters = max_iters,
                                                                            max_attempts = max_iters,
                                                                            curve = True,
                                                                            random_state = seed,
                                                                            )
    end = time.time()
    solvetime_rh = round(end - start,3)
    vPrint(f'RHC:\n\tBest Fit: {best_fitness_rhc}\n\tSolve Time: {solvetime_rh}')
    plt.plot(f*fitness_curve_rhc[:,0],'b-',label='RHC')
    # Simulated Annealing
    start = time.time()
    best_state_sa, best_fitness_sa, fitness_curve_sa = simulated_annealing(problem = problem,
                                                                           schedule = schedule,
                                                                max_iters = max_iters,
                                                                max_attempts = max_iters,
                                                                curve = True,
                                                                random_state = seed,
                                                                )
    end = time.time()
    solvetime_sa = round(end - start,3)
    vPrint(f'SA:\n\tBest Fit: {best_fitness_sa}\n\tSolve Time: {solvetime_sa}')
    plt.plot(f*fitness_curve_sa[:,0],'r-',label='SA')
    
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
    plt.plot(f*fitness_curve_ga[:,0],'g-',label='GA')
    
    # MIMIC
    start = time.time()
    best_state_m, best_fitness_m, fitness_curve_m = mimic(problem = problem,
                                                    pop_size = pop_size,
                                                    max_iters = max_iters,
                                                    max_attempts = int(max_iters/4),
                                                    curve = True,
                                                    random_state = seed,
                                                    )
    end = time.time()
    solvetime_m = round(end - start,3)
    vPrint(f'MIMIC:\n\tBest Fit: {best_fitness_m}\n\tSolve Time: {solvetime_m}')
    plt.plot(f*fitness_curve_m[:,0],'k-',label='MIMIC')
    
    # Format Chart
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'Images/{title}_Seed{seed}_Comparision.png')
    plt.show()
    
    # Plot solve times
    fig = plt.subplots(figsize=(12,10),dpi = 200)
    plt.bar('RHC',solvetime_rh,log=True)
    plt.bar('SA',solvetime_sa,log=True)
    plt.bar('GA',solvetime_ga,log=True)
    plt.bar('MIMIC',solvetime_m,log=True)
    plt.ylabel('Solution Time (sec)')
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'Images/{title}_Seed{seed}_SolveTime.png')
    plt.show()
    
###############################################################################
# Travelling Sales Optimization Problem
###############################################################################
# =============================================================================
# experiment_name = 'TSP-RHC'
# problem = TSPGenerator.generate(seed=seed, number_of_cities=100)
# 
# rhc = RHCRunner(problem=problem,
#                 experiment_name=experiment_name,
#                 output_directory=os.getcwd(),
#                 seed=seed,
#                 iteration_list=2 ** np.arange(10),
#                 max_attempts=5000,
#                 restart_list=[25, 75, 100])   
# 
# # the two data frames will contain the results
# df_run_stats, df_run_curves = rhc.run()      
# =============================================================================

print('######################################################################')
print('Travelling Salesman Problem')
print('######################################################################')

n_cities = [10,50,100,]
for n in n_cities:
    for seed in seeds:
        probConnect = 0.5
        useDistances = True
        
        # Create a connected graph to represent cities and routes
        
        g = nx.erdos_renyi_graph(n, probConnect, seed=seed, directed=False)
        while not nx.is_connected(g):
            n += 1
            vPrint('Graph not connected, increasing n...')
            g = nx.erdos_renyi_graph(n, 0.5, seed=seed, directed=False)
        vPrint(f'Graph set to {n} nodes, is connected.')
        vPrint(f'\tEdges = {len(g.edges)}')
        # Add weights
        for (u, v) in g.edges():
            g.edges[u,v]['weight'] = np.ceil(random.random()*10)
            
        distances = [(u,v,w['weight']) for (u,v,w) in g.edges(data=True)]
        
        
        ## Create distance matrix
        dist_matrix = np.ones((n,n))
        for i in range(n):
            for ii in range(n):
                edge = (i,ii)
                if edge in distances:
                    dist_matrix[i][ii] = -1
                else:
                    dist_matrix[i][ii] = 1
        
        # Compute held_karp
    # =============================================================================
    #     start = time.time()
    #     brute_force = hk.held_karp(dist_matrix)
    #     end = time.time()
    #     brute_time = end - start
    #     brute_path = []
    #     for n, target in enumerate(brute_force[1]):
    #         source = brute_force[1][n-1]
    #         if source not in brute_path:
    #             brute_path.append(source)
    #             
    #         shortest_p = nx.shortest_path(g,source,target)
    #         for n_ in shortest_p:
    #             if n_ != source and n_ != target:
    #                 brute_path.append(n)
    #                 
    #     print(f'Brute force path: {brute_path}')
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
                'keep_pct':keep_pct,
                  }
        
        title = f'TravellingSales_{n}_seed{seed}'
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
#             'keep_pct':keep_pct,
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

bits = 50
n_bits = 4
mask = np.random.randint(0,n_bits,bits)
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
            total_value -= 0
            total_value = max(total_value,0)
    for i, s in enumerate(state[::-1]):
        if s == mask[-i]:
            m += 1
            total_value += m
        else:
            m = 0
            total_value -= 0
            total_value = max(total_value,0)
    
    
    
    return total_value

optimal_val = random_bit_match(mask,mask)
print(f'Optimal Random Bit Match = {optimal_val}')

kwargs = {'mask':mask,}

fitness = CustomFitness(fitness_fn = random_bit_match, problem_type='discrete', **kwargs)

problem = DiscreteOpt(length = bits, fitness_fn = fitness, maximize=True, max_val=n_bits)

##################################
# Solve using each algorithm
##################################
kwargs = {
        'restarts':restarts,
        'max_iters':max_iters,
        'max_attempts':max_attempts,
        'pop_size':bits,
        'mutation_prob':mutation_prob,
        'keep_pct':0.1,
          }

# Find optimal
vPrint('Finding optimal mimic for RBM problem...')
experiment_name = 'RandomBitMatch'
iteration_list = list(map(int,np.linspace(10,1000,5)))
population_sizes = list(map(int,np.linspace(10,500,5)))
mmc = MIMICRunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=os.getcwd(),
                  population_sizes = population_sizes,
                  seed=seed,
                  iteration_list = iteration_list,
                  max_attempts=500,
                  keep_percent_list=[0.05,0.1,0.2, 0.5, 0.75],
                  use_fast_mimic = True)
run_start = time.time()
df_run_stats, df_run_curves = mmc.run()
run_end = time.time()
vPrint('MIMIC runner time for RBC: {run_end - run_start}')

best_mimic_run = mmc.run_stats_df.iloc[np.argmax(mmc.run_stats_df['Fitness']),:]
best_pop_size = best_mimic_run['Population Size']
best_iters = best_mimic_run['max_iters']
best_state = best_mimic_run['State']
best_keep_pct = best_mimic_run['Keep Percent']
best_state = list(map(float,best_state.replace(']','').replace('[','').replace(' ','').split(',')))

kwargs['max_iters'] = best_iters
kwargs['keep_pct'] = best_keep_pct
kwargs['pop_size'] = best_pop_size

runProblem(problem=problem,title='RandomBitMatch',**kwargs)


###############################################################################
# Knapsack Problem
###############################################################################
print('######################################################################')
print('Four Peaks Problem')
print('######################################################################')
bits = 100
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
        'keep_pct':keep_pct,
          }

runProblem(problem=problem,title='FourPeaks',**kwargs)


###############################################################################
# Knapsack Problem
###############################################################################
print('######################################################################')
print('Knapsack Problem')
print('######################################################################')
n_items = 50
weights = [0] # First item has zero weight and zero value
values = [0]
weights.extend([random.randint(1,n_items) for _ in range(n_items - 1)])
values.extend([random.randint(0,n_items) for _ in range(n_items - 1)])
max_weight_pct = 0.75
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
        'keep_pct':keep_pct,
          }

runProblem(problem=problem,title='Knapsack',**kwargs)


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
Xdata = ds.iloc[:,:-1]
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

## Create sample weights for scoring which penalize false positives (i.e. classifying
##  a poisonuous mushroom as edible)
sample_weights = [2 if i == 1 else 1 for i in Ytest]

############################
# Neural Network Model - run each optimization algorithm
############################
loss_fns = [skm.log_loss,skm.mean_squared_error,skm.mean_squared_log_error]

loss_fn = loss_fns[1]

algos = ['random_hill_climb', 'simulated_annealing','genetic_alg','gradient_descent']
actives = ['identity', 'relu', 'sigmoid','tanh']
mutation_probs = np.linspace(0.001,0.1,10)
h_nodes = [[int(x)] for x in np.linspace(5, 25,5)]
pop_sizes = list(map(int,np.linspace(50,500,5)))
learning_rate = 0.05
## Run model for each algorithm
_rhc_score = []
_sa_score = []
_ga_score = []
_rhc_score_f1 = []
_sa_score_f1 = []
_ga_score_f1 = []

max_iters_ = list(map(int,np.linspace(10,1000,10)))

for max_iters in max_iters_: 
    #RHC
    algo_rhc = algos[0]
    model_rhc = NeuralNetwork(algorithm = algo_rhc,
                          activation = 'relu',
                          hidden_nodes = [Xdata.shape[1]],
                          mutation_prob = 0.01,
                          max_iters = max_iters,
                          learning_rate = learning_rate,
                          curve = True, 
                          random_state = seed,
                          loss_fn = loss_fn)
    
    start = time.time()
    model_rhc.fit(Xtrain,Ytrain)
    end = time.time()
    fit_time = end - start
    score = model_rhc.score(Xtest,Ytest,sample_weights)
    y_pred = model_rhc.predict(Xtest)
    f1 = f1_score(Ytest,y_pred)
    vPrint(f'Algorithm: {algo_rhc}\n\tScore = {score}\n\tF1 = {f1}\n\tFit Time = {fit_time}')
    _rhc_score.append(score)
    _rhc_score_f1.append(score)
    
    #SA
    algo_sa = algos[1]
    model_sa = NeuralNetwork(algorithm = algo_sa,
                          activation = 'relu',
                          hidden_nodes = [Xdata.shape[1]],
                          mutation_prob = 0.01,
                          max_iters = max_iters,
                          learning_rate = learning_rate,
                          curve = True, 
                          random_state = seed,
                          loss_fn = loss_fn)
    start = time.time()
    model_sa.fit(Xtrain,Ytrain)
    end = time.time()
    fit_time = end - start
    score = model_sa.score(Xtest,Ytest,sample_weights)
    y_pred = model_sa.predict(Xtest)
    f1 = f1_score(Ytest,y_pred)
    vPrint(f'Algorithm: {algo_sa}\n\tScore = {score}\n\tF1 = {f1}\n\tFit Time = {fit_time}')
    _sa_score.append(score)
    _sa_score_f1.append(score)
    
    #GA
    algo_ga = algos[2]
    model_ga = NeuralNetwork(algorithm = algo_ga,
                          activation = 'relu',
                          hidden_nodes = [Xdata.shape[1]],
                          mutation_prob = mutation_prob,
                          max_iters = max_iters,
                          learning_rate = learning_rate,
                          pop_size = pop_size,
                          curve = True, 
                          random_state = seed,
                          loss_fn = loss_fn)
    start = time.time()
    model_ga.fit(Xtrain,Ytrain)
    end = time.time()
    fit_time = end - start
    score = model_ga.score(Xtest,Ytest,sample_weights)
    y_pred = model_ga.predict(Xtest)
    f1 = f1_score(Ytest,y_pred)
    vPrint(f'Algorithm: {algo_ga}\n\tScore = {score}\n\tF1 = {f1}\n\tFit Time = {fit_time}')
    _ga_score.append(score)
    _ga_score_f1.append(score)
    
    # =============================================================================
    # #GD
    # algo_gd = algos[3]
    # max_iters = 1000
    # model_gd = NeuralNetwork(algorithm = algo_gd,
    #                       activation = 'relu',
    #                       hidden_nodes = [Xdata.shape[1]],
    #                       mutation_prob = 0.01,
    #                       max_iters = max_iters,
    #                       learning_rate = learning_rate,
    #                       curve = True, 
    #                       random_state = seed,
    #                       loss_fn = loss_fn)
    # start = time.time()
    # model_gd.fit(Xtrain,Ytrain)
    # end = time.time()
    # fit_time = end - start
    # score = model_gd.score(Xtest,Ytest)
    # y_pred = model_gd.predict(Xtest)
    # f1 = f1_score(Ytest,y_pred)
    # vPrint(f'Algorithm: {algo_gd}\n\tScore = {score}\n\tF1 = {f1}\n\tFit Time = {fit_time}')
    # 
    # =============================================================================
scaler = preprocessing.MinMaxScaler()

_rhc = -1*model_rhc.fitness_curve[:,0]
_sa = -1*model_sa.fitness_curve[:,0]
_ga = -1*model_ga.fitness_curve[:,0]
# =============================================================================
# _rhc = scaler.fit_transform(model_rhc.fitness_curve[:,0].reshape(-1,1))
# _sa = scaler.fit_transform(model_sa.fitness_curve[:,0].reshape(-1,1))
# _ga = scaler.fit_transform(model_ga.fitness_curve[:,0].reshape(-1,1))
# 
# =============================================================================
## Initialize plot
fig, ax = plt.subplots(figsize = (12,10),dpi = 200)## Add data to plot
ax.plot(_rhc,label=f'{algo_rhc}')
## Add data to plot
ax.plot(_sa,label=f'{algo_sa}')
## Add data to plot
ax.plot(_ga,label=f'{algo_ga}')
###### FINISH PLOT
plt.grid()
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('Images/NN_optimization.png')
plt.show()




main_end = time.time()
main_time = main_end - main_start
main_time = round(main_time,2)
print(f'Total run time: {main_time} seconds')


