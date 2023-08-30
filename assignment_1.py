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
# Load data sets
random.seed(1)

if __name__ == '__main__':
    os.chdir(os.path.split(__file__)[0])
    
creditDefaultFile = r".\data\default of credit card clients.xls"

ds1 = pd.read_excel(creditDefaultFile,skiprows=1)

###############################################################################
# Decision Tree Test
###############################################################################
dt_out_scores_random = []
dt_in_scores_random = []
dt_out_scores_best = []
dt_in_scores_best = []

min_samples = np.linspace(200,2,10)

## Get X, Y data for test and train
Xdata = ds1.iloc[:,1:-1]
Ydata = ds1.iloc[:,-1]

# Split data
test_size = 0.2

Xtrain, Xtest, Ytrain, Ytest = ms.train_test_split(Xdata, Ydata, test_size=test_size)
for mss in min_samples:
    ## Create random decision tree 
    min_samples_leaf = int(mss)
    max_depth = None
    ccp_alpha = 0.0
    
    dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
                                splitter='random')
    # Print attributes
    print(f'\tRandom Decision Tree min_samples_leaf: {dt.min_samples_leaf}')
    print(f'\tRandom Decision Tree max depth: {dt.max_depth}')
    
    # Fit the dt to training data
    dt.fit(Xtrain,Ytrain)
    
    
    # Score on test data
    dtscore = dt.score(Xtest,Ytest)
    
    print(f'\tRandom Decision Tree Score: {dtscore}')
    
    # Add to scores
    dt_out_scores_random.append(dtscore)
    dt_in_scores_random.append(dt.score(Xtrain,Ytrain))
    
    
    ## Decision tree with best split
    dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
                                splitter='best')
    # Print attributes
    print(f'\tBestSplit Decision Tree min_samples_leaf: {dt.min_samples_leaf}')
    print(f'\tBestSplit Decision Tree max depth: {dt.max_depth}')
    
    # Fit the dt to training data
    dt.fit(Xtrain,Ytrain)
    
    
    # Score on test data
    dtscore = dt.score(Xtest,Ytest)
    
    print(f'\tBestSplit Decision Tree Score: {dtscore}')
    
    # Add to scores
    dt_out_scores_best.append(dtscore)
    dt_in_scores_best.append(dt.score(Xtrain,Ytrain))
    
## Create plot
fig, ax =  plt.subplots(1,2,figsize=(6,4),dpi=200)

#Random Split
ax[0].plot(min_samples,dt_in_scores_random,label='Training Score')
ax[0].plot(min_samples,dt_out_scores_random,label='Test Score')
    
ax[0].set_xlabel('min_samples_leaf')
ax[0].set_ylabel('Score')
ax[0].grid()
ax[0].legend()

# Best Split
ax[1].plot(min_samples,dt_in_scores_best,label='Training Score')
ax[1].plot(min_samples,dt_out_scores_best,label='Test Score')
    
ax[1].set_xlabel('min_samples_leaf')
ax[1].set_ylabel('Score')
ax[1].grid()
ax[1].legend()

plt.savefig('DecisionTreeClassifier_Figure.png')
plt.show() # Save fig

    
##### Decision Tree with Boosting
dt_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                      max_depth=1, random_state=0)

# Fit dt_boost
    
    
    
    
    
    
    







