## CS7641 Machine Learning - Assignment 1
"""
    Create models for Decision Tree,DT with boosting, SVM, KNN, Neural Nets 





"""
# Simple Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing, utils
import sklearn.model_selection as ms
# Decision Tree with boosting
from sklearn.ensemble import GradientBoostingClassifier

import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Load data sets
seed = 1

if __name__ == '__main__':
    os.chdir(os.path.split(__file__)[0])
    
creditDefaultFile = r".\data\default of credit card clients.xls"

ds1 = pd.read_excel(creditDefaultFile,skiprows=1)

###############################################################################
# Helper Functions
###############################################################################

def vPrint(text: str = '', verbose: bool = True):
    if verbose:
        print(text)

def timefit(model,args,verbose: bool = True):
    start = time()
    model.fit(*args)
    end = time()
    inter = end - start
    st = '{:,.6}'.format(inter)
    t = f'{model}: Fit Time: {st} seconds'
    vPrint(t,verbose)
    return inter

###############################################################################
# Decision Tree Test
###############################################################################
dt_out_scores_random = []
dt_in_scores_random = []
dt_out_scores_best = []
dt_in_scores_best = []
dt_rand_time = []
dt_best_time = []

# Min samples to try
min_samples = list(map(int,np.linspace(200,2,10)))
# Max depth to try
max_depths = list(map(int,np.linspace(1,20,10)))

## Get X, Y data for test and train
Xdata = ds1.iloc[:,1:-1]
Ydata = ds1.iloc[:,-1]

# Split data (set random seed)
test_size = 0.2
Xtrain, Xtest, Ytrain, Ytest = ms.train_test_split(Xdata,
                                                   Ydata,
                                                   test_size=test_size,
                                                   random_state=seed)

## Loop for min samples
for mss in min_samples:
    ## Create random decision tree 
    min_samples_leaf = int(mss)
    max_depth = None
    ccp_alpha = 0.0
    
    dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
                                splitter='random',
                                random_state = seed)
    # Print attributes
    print(f'\tRandom Decision Tree min_samples_leaf: {dt.min_samples_leaf}')
    print(f'\tRandom Decision Tree max depth: {dt.max_depth}')
    
    # Fit the dt to training data
    t = timefit(dt,(Xtrain,Ytrain))
    dt_rand_time.append(t)
    
    # Score on test data
    dtscore = dt.score(Xtest,Ytest)
    
    print(f'\tRandom Decision Tree Score: {dtscore}')
    
    # Add to scores
    dt_out_scores_random.append(dtscore)
    dt_in_scores_random.append(dt.score(Xtrain,Ytrain))
    
    
    ## Decision tree with best split
    dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
                                splitter='best',
                                random_state = seed)
    # Print attributes
    print(f'\tBestSplit Decision Tree min_samples_leaf: {dt.min_samples_leaf}')
    print(f'\tBestSplit Decision Tree max depth: {dt.max_depth}')
    
    # Fit the dt to training data
    t = timefit(dt,(Xtrain,Ytrain))
    dt_best_time.append(t)
    
    # Score on test data
    dtscore = dt.score(Xtest,Ytest)
    
    print(f'\tBestSplit Decision Tree Score: {dtscore}')
    
    # Add to scores
    dt_out_scores_best.append(dtscore)
    dt_in_scores_best.append(dt.score(Xtrain,Ytrain))

## Loop for max depth, fixed min_samples_leaf = 50
dt_out_scores_random_md = []
dt_in_scores_random_md = []
dt_out_scores_best_md = []
dt_in_scores_best_md = []
dt_rand_time_md = []
dt_best_time_md = []

for md in max_depths:
    ## Create random decision tree 
    min_samples_leaf = 100
    max_depth = md
    ccp_alpha = 0.0
    
    dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,max_depth=max_depth,
                                splitter='random',
                                random_state = seed)
    # Print attributes
    print(f'\tRandom Decision Tree min_samples_leaf: {dt.min_samples_leaf}')
    print(f'\tRandom Decision Tree max depth: {dt.max_depth}')
    
    # Fit the dt to training data
    t = timefit(dt,(Xtrain,Ytrain))
    dt_rand_time_md.append(t)
    
    # Score on test data
    dtscore = dt.score(Xtest,Ytest)
    
    print(f'\tRandom Decision Tree Score: {dtscore}')
    
    # Add to scores
    dt_out_scores_random_md.append(dtscore)
    dt_in_scores_random_md.append(dt.score(Xtrain,Ytrain))
    
    
    ## Decision tree with best split
    dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,max_depth=max_depth,
                                splitter='best',
                                random_state = seed)
    # Print attributes
    print(f'\tBestSplit Decision Tree min_samples_leaf: {dt.min_samples_leaf}')
    print(f'\tBestSplit Decision Tree max depth: {dt.max_depth}')
    
    # Fit the dt to training data
    timefit(dt,(Xtrain,Ytrain))
    dt_best_time_md.append(t)
    
    # Score on test data
    dtscore = dt.score(Xtest,Ytest)
    
    print(f'\tBestSplit Decision Tree Score: {dtscore}')
    
    # Add to scores
    dt_out_scores_best_md.append(dtscore)
    dt_in_scores_best_md.append(dt.score(Xtrain,Ytrain))
    
## Create plot
fig, ax =  plt.subplots(2,2,figsize=(12,6),dpi=200)
## Plot fit times on second axis for each tree

#Random Split Min_samples
ax00 = ax[0][0].twinx()
l1 = ax[0][0].plot(min_samples,dt_in_scores_random,label='Training Score')
l2 = ax[0][0].plot(min_samples,dt_out_scores_random,label='Test Score')
l3 = ax00.plot(min_samples,dt_rand_time,'k--',label='Training Time')
ax00.set_ylabel('Training Time')
ls = l1+ l2 + l3
lb = [l.get_label() for l in ls]
ax[0][0].set_xlabel('min_samples_leaf')
ax[0][0].set_ylabel('Score')
ax[0][0].grid()
ax[0][0].legend(ls,lb,loc=0)

# Best Split Min_samples
ax01 = ax[0][1].twinx()
l1 = ax[0][1].plot(min_samples,dt_in_scores_best,label='Training Score')
l2 = ax[0][1].plot(min_samples,dt_out_scores_best,label='Test Score')
l3 = ax01.plot(min_samples,dt_best_time,'k--',label = 'Training Time')
ax[0][1].set_xlabel('min_samples_leaf')
ax01.set_ylabel('Training Time')
ax[0][1].set_ylabel('Score')
ax[0][1].grid()
ls = l1 + l2 + l3
lb = [l.get_label() for l in ls]
ax[0][1].legend(ls,lb,loc=0)

#Random Split Max_depths
ax10 = ax[1][0].twinx()
l1 = ax[1][0].plot(max_depths,dt_in_scores_random_md,label='Training Score')
l2 = ax[1][0].plot(max_depths,dt_out_scores_random_md,label='Test Score')
l3 = ax10.plot(max_depths,dt_rand_time_md,'k--',label='Training Time')
ax10.set_ylabel('Training Time')
ls = l1 + l2 + l3
lb = [l.get_label() for l in ls]
ax[1][0].set_xlabel(f'Max_depth (min_samples={min_samples_leaf})')
ax[1][0].set_ylabel('Score')
ax[1][0].grid()
ax[1][0].legend(ls,lb,loc=0)

# Best Split Max_depths
ax11 = ax[1][1].twinx()
l1 = ax[1][1].plot(max_depths,dt_in_scores_best_md,label='Training Score')
l2 = ax[1][1].plot(max_depths,dt_out_scores_best_md,label='Test Score')
l3 = ax11.plot(max_depths,dt_best_time_md,'k--',label='Training Time')
ax11.set_ylabel('Training Time')
ls = l1 + l2 + l3
lb = [l.get_label() for l in ls]
ax[1][1].set_xlabel(f'Max_depth (min_samples={min_samples_leaf})')
ax[1][1].set_ylabel('Score')
ax[1][1].grid()
ax[1][1].legend(ls,lb,loc=0)

ax[0][0].set_title('Random Splitter')
ax[0][1].set_title('Best Feature Splitter')

fig.tight_layout()
plt.suptitle('Decision Tree Classifier Hyperparameter Test')
plt.savefig('DecisionTreeClassifier_Figure.png')
plt.show() # Save fig

    
##### Decision Tree with Boosting
n_estimators = 100
dt_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                      max_depth=1, random_state=0)

# Fit dt_boost
    
    
    
    
    
    
    







