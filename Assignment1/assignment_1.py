## CS7641 Machine Learning - Assignment 1
"""
    Create models for Decision Tree,DT with boosting, SVM, KNN, Neural Nets 





"""
import multiprocessing
import argparse

# Simple Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from  sklearn import model_selection

from sklearn.decomposition import PCA
# Decision Tree with boosting
from sklearn.ensemble import GradientBoostingClassifier
# Neural Network MLP
from sklearn.neural_network import MLPClassifier
# KNN
from sklearn.neighbors import KNeighborsClassifier
# Support Vector Machine - SVM
from sklearn.linear_model import SGDClassifier

import os
import pandas as pd
#simport random
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time


verbose = True
###############################################################################
# Helper Functions
###############################################################################

def vPrint(text: str = '', verbose: bool = verbose):
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

def timescore(model,args,verbose: bool = True):
    start = time()
    score = model.score(*args)
    end = time()
    inter = end - start
    st = '{:,.6}'.format(inter)
    t = f'{model}: Score Time: {st} seconds'
    vPrint(t,verbose)
    return inter, score

def processData(Xdata):
    ## PRE-PROCESS ALL DATA
    for col in range(Xdata.shape[1]):
        le = preprocessing.LabelEncoder()
        Xdata.iloc[:,col] = le.fit_transform(Xdata.iloc[:,col])
    
    # Only include data that includes 95% of data
    min_exp_ = 0.95
    pca = PCA()
    pca.fit_transform(Xdata)
    i = -1
    exp_ = 0
    while exp_ < min_exp_:
        i += 1
        # *** PCA explained variance is sorted ***
        exp_ += pca.explained_variance_[i] / sum(pca.explained_variance_)
        
    # Only include up to the i-th attribute
    Xdata = Xdata.iloc[:,:i+1]
    
    return Xdata

# Load data sets
seed = 903860493

if __name__ == '__main__':
    os.chdir(os.path.split(__file__)[0])
# =============================================================================
#     
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-v','--verbose',type=bool,default=True,dest='verbose',required=False)
#     
# =============================================================================
    
## Select Data Sets
creditDefaultFile = r"data/default of credit card clients.xls"
roomOccupancyFile = r"data/Occupancy_Estimation.csv"
mushroomClassificationFile = r"data\secondary+mushroom+dataset\MushroomDataset\secondary_data.csv"
studentDropoutfile = r"data\predict+students+dropout+and+academic+success\data.csv"

creditDefaultFile = os.path.join(os.getcwd(),creditDefaultFile)
roomOccupancyFile = os.path.join(os.getcwd(),roomOccupancyFile)

ds1 = pd.read_excel(creditDefaultFile,skiprows=1)
ds2 = pd.read_csv(roomOccupancyFile)
ds3 = pd.read_csv(mushroomClassificationFile)
ds4 = pd.read_csv(studentDropoutfile)

datasets = {
    'Credit Default':ds1,
    #'Room Occupancy':ds2,
    'Mushroom Classification':ds3,
    #'Student Dropout':ds4,
    }

## ** DS1 is sparse with defaulters so lets randomly balance
ds1_p = random.choices(ds1.loc[(ds1['default payment next month'] == 0)].index,k=len(ds1.loc[(ds1['default payment next month'] == 1)]))
ds1_d = ds1.loc[(ds1['default payment next month'] == 1)].index
# =============================================================================
# 
# ds1_i = []
# ds1_i.extend(list(ds1_p))
# ds1_i.extend(list(ds1_d))
# 
# # Reset ds1.index 
# ds1 = ds1.iloc[ds1_i]
# datasets['Credit Default'] = ds1
# =============================================================================

#Train Test Split for all experiments 
test_size = 0.2

# Verbosity level
run_grid_search = True

# Run n_jobs in parallel
n_jobs = -14

# Savfigure flag
savefig = True

cores = min(n_jobs,os.cpu_count()) if n_jobs > 0 else max(1,os.cpu_count() + n_jobs) 
cores_ = f'{cores} cores' if cores > 1 else '1 core'
vPrint(f'Using {cores_} to process')


################0###############################################################
# Decision Tree Test
# ****Add Pruning*****
###############################################################################

# Min samples to try
min_samples = list(map(int,np.linspace(200,2,10)))
# Max depth to try
max_depths = list(map(int,np.linspace(1,20,10)))
# Vary ccp_alpha
ccp_alphas = list(map(float,np.linspace(0,0.01,10)))

# =============================================================================
# def experiment1():
# =============================================================================
vPrint('Starting experiment 1')
#### Run test for each data set
for DataSetName, ds in datasets.items():

        
    ## Get X, Y data for test and train
    Xdata = ds.iloc[:,1:-1]
    Ydata = ds.iloc[:,-1]
    
    ## PRE-PROCESS ALL DATA
    
    for col in range(Xdata.shape[1]):
        le = preprocessing.LabelEncoder()
        scaler = preprocessing.MinMaxScaler()
        Xdata.iloc[:,col] = le.fit_transform(Xdata.iloc[:,col])
# =============================================================================
#         Xdata.iloc[:,col] = scaler.fit_transform(np.array(Xdata.iloc[:,col]).reshape(-1,1))
# =============================================================================
    
    pca = PCA()
    pca.fit_transform(Xdata)
    best_features = pca.explained_variance_
    scaler = preprocessing.MinMaxScaler()
    best_features = scaler.fit_transform(best_features.reshape(-1,1))
    
    
# =============================================================================
#     
#     # Only include data that includes 95% of data
#     min_exp_ = 0.95
#     pca = PCA()
#     pca.fit_transform(Xdata)
#     i = -1
#     exp_ = 0
#     while exp_ < min_exp_:
#         i += 1
#         # *** PCA explained variance is sorted ***
#         exp_ += pca.explained_variance_[i] / sum(pca.explained_variance_)
#         
#     # Only include up to the i-th attribute
#     Xdata = Xdata.iloc[:,:i]
# =============================================================================
        
    
    # Split data (set random seed)
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(Xdata,
                                                       Ydata,
                                                       test_size=test_size,
                                                       random_state=seed)
    
    dt_out_scores_random = []
    dt_in_scores_random = []
    dt_out_scores_best = []
    dt_in_scores_best = []
    dt_rand_time = []
    dt_best_time = []
    
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
        vPrint(f'\tRandom Decision Tree min_samples_leaf: {dt.min_samples_leaf}')
        vPrint(f'\tRandom Decision Tree max depth: {dt.max_depth}')
        
        # Fit the dt to training data
        t = timefit(dt,(Xtrain,Ytrain))
        dt_rand_time.append(t)
        
        # Score on test data
        dtscore = dt.score(Xtest,Ytest)
        
        vPrint(f'\tRandom Decision Tree Score: {dtscore}')
        
        # Add to scores
        dt_out_scores_random.append(dtscore)
        dt_in_scores_random.append(dt.score(Xtrain,Ytrain))
        
        
        ## Decision tree with best split
        dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
                                    splitter='best',
                                    random_state = seed)
        # Print attributes
        vPrint(f'\tBestSplit Decision Tree min_samples_leaf: {dt.min_samples_leaf}')
        vPrint(f'\tBestSplit Decision Tree max depth: {dt.max_depth}')
        
        # Fit the dt to training data
        t = timefit(dt,(Xtrain,Ytrain))
        dt_best_time.append(t)
        
        # Score on test data
        dtscore = dt.score(Xtest,Ytest)
        
        vPrint(f'\tBestSplit Decision Tree Score: {dtscore}')
        
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
        vPrint(f'\tRandom Decision Tree min_samples_leaf: {dt.min_samples_leaf}')
        vPrint(f'\tRandom Decision Tree max depth: {dt.max_depth}')
        
        # Fit the dt to training data
        t = timefit(dt,(Xtrain,Ytrain))
        dt_rand_time_md.append(t)
        
        # Score on test data
        dtscore = dt.score(Xtest,Ytest)
        
        vPrint(f'\tRandom Decision Tree Score: {dtscore}')
        
        # Add to scores
        dt_out_scores_random_md.append(dtscore)
        dt_in_scores_random_md.append(dt.score(Xtrain,Ytrain))
        
        
        ## Decision tree with best split
        dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,max_depth=max_depth,
                                    splitter='best',
                                    random_state = seed)
        # Print attributes
        vPrint(f'\tBestSplit Decision Tree min_samples_leaf: {dt.min_samples_leaf}')
        vPrint(f'\tBestSplit Decision Tree max depth: {dt.max_depth}')
        
        # Fit the dt to training data
        timefit(dt,(Xtrain,Ytrain))
        dt_best_time_md.append(t)
        
        # Score on test data
        dtscore = dt.score(Xtest,Ytest)
        
        vPrint(f'\tBestSplit Decision Tree Score: {dtscore}')
        
        # Add to scores
        dt_out_scores_best_md.append(dtscore)
        dt_in_scores_best_md.append(dt.score(Xtrain,Ytrain))
        
    dt_out_scores_random_ccp = []
    dt_in_scores_random_ccp = []
    dt_out_scores_best_ccp = []
    dt_in_scores_best_ccp = []
    dt_best_time_ccp = []
    dt_rand_time_ccp = []
    
    for ccp_alpha_ in ccp_alphas:
        min_samples_leaf = 2
        max_depth = 20
        model = DecisionTreeClassifier(splitter='best',
                                    random_state = seed,
                                    max_depth = max_depth,
                                    min_samples_leaf = min_samples_leaf,
                                    ccp_alpha=ccp_alpha_)
        
        dt_best_time_ccp.append(timefit(model,(Xtrain,Ytrain)))
        dt_out_scores_best_ccp.append(model.score(Xtest,Ytest))
        dt_in_scores_best_ccp.append(model.score(Xtrain,Ytrain))
        
        model = DecisionTreeClassifier(splitter='random',
                                    random_state = seed,
                                    ccp_alpha=ccp_alpha_)
        
        dt_rand_time_ccp.append(timefit(model,(Xtrain,Ytrain)))
        dt_out_scores_random_ccp.append(model.score(Xtest,Ytest))
        dt_in_scores_random_ccp.append(model.score(Xtrain,Ytrain))
        
        
        
    #######################
    # Create plot
    #######################
    fig, ax =  plt.subplots(3,2,figsize=(12,10),dpi=200)
    ## Plot fit times on second axis for each tree
    
    #Random Split Min_samples
    ax00 = ax[0][0].twinx()
    l1 = ax[0][0].plot(min_samples,dt_in_scores_random,label='Training Score')
    l2 = ax[0][0].plot(min_samples,dt_out_scores_random,label='Test Score')
    l3 = ax00.plot(min_samples,dt_rand_time,'k--',label='Training Time')
    ax00.set_ylabel('Training Time (seconds)')
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
    ax01.set_ylabel('Training Time (seconds)')
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
    ax10.set_ylabel('Training Time (seconds)')
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
    ax11.set_ylabel('Training Time (seconds)')
    ls = l1 + l2 + l3
    lb = [l.get_label() for l in ls]
    ax[1][1].set_xlabel(f'Max_depth (min_samples={min_samples_leaf})')
    ax[1][1].set_ylabel('Score')
    ax[1][1].grid()
    ax[1][1].legend(ls,lb,loc=0)
    
    ax[0][0].set_title('Random Splitter')
    ax[0][1].set_title('Best Feature Splitter')
    
    # Random Split ccp_alphas
    ax20 = ax[2][0].twinx()
    l1 = ax[2][0].plot(ccp_alphas,dt_in_scores_random_ccp,label='Training Score')
    l2 = ax[2][0].plot(ccp_alphas,dt_out_scores_random_ccp,label='Test Score')
    l3 = ax20.plot(ccp_alphas,dt_rand_time_ccp,'k--',label='Training Time')
    ax20.set_ylabel('Training Time (seconds)')
    ls = l1 + l2 + l3
    lb = [l.get_label() for l in ls]
    ax[2][0].set_xlabel(f'CCP_Alpha')
    ax[2][0].set_ylabel('Score')
    ax[2][0].grid()
    ax[2][0].legend(ls,lb,loc=0)
    
    # Best Split ccp_alphas
    ax21 = ax[2][1].twinx()
    l1 = ax[2][1].plot(ccp_alphas,dt_in_scores_best_ccp,label='Training Score')
    l2 = ax[2][1].plot(ccp_alphas,dt_out_scores_best_ccp,label='Test Score')
    l3 = ax21.plot(ccp_alphas,dt_best_time_ccp,'k--',label='Training Time')
    ax21.set_ylabel('Training Time (seconds)')
    ls = l1 + l2 + l3
    lb = [l.get_label() for l in ls]
    ax[2][1].set_xlabel(f'CCP_Alpha')
    ax[2][1].set_ylabel('Score')
    ax[2][1].grid()
    ax[2][1].legend(ls,lb,loc=0)
    

###############################################################################
# Setup GridSeachCV for Decision Tree with Pruning for each dataset
###############################################################################
    if run_grid_search:
        estimator = DecisionTreeClassifier(random_state = seed) 
        vPrint(f'Starting parameter grid search for {estimator}: {DataSetName}\n')
        param_grid = {
            'min_samples_leaf':min_samples,
            'max_depth':max_depths,
            'ccp_alpha':ccp_alphas,
            'splitter':['best','random'],
            }
        
        fold = []
        for i in range(Xdata.shape[0]):
            if Xdata.index[i] in Xtrain.index:
                fold.append(-1)
            else:
                fold.append(0)
                
        ps = model_selection.PredefinedSplit(fold) # Fix the fold on the train data
        
        clf = model_selection.GridSearchCV(estimator=estimator, 
                                           param_grid = param_grid, 
                                           verbose=3 if verbose else 0,
                                           scoring = 'accuracy',
                                           n_jobs = n_jobs,
                                           return_train_score = True,
                                           cv=ps,
                                           )
    
        clf.fit(Xdata,Ydata)
    
        vPrint(f'GridSearchCV Complete for {DataSetName} using {estimator}.')
        
        clf.score(Xtest,Ytest)
        clf.score(Xtrain,Ytrain)
        
        best_estimator_ = clf.best_estimator_
        best_estimator_.fit(Xtrain,Ytrain)

        # Add to plot
        ax[0][0].plot(best_estimator_.min_samples_leaf,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[0][1].plot(best_estimator_.min_samples_leaf,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[1][0].plot(best_estimator_.max_depth,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[1][1].plot(best_estimator_.max_depth,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[2][0].plot(best_estimator_.ccp_alpha,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[2][1].plot(best_estimator_.ccp_alpha,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        
    
    fig.tight_layout()
    plt.suptitle(f'Decision Tree - {DataSetName}\n')
    if savefig:
        plt.savefig(f'Images/DecisionTreeClassifier_{DataSetName}_Figure.png')
    plt.show() # Save fig
    
###############################################################################
# Decision Tree with boosting using GBC
###############################################################################
n_estimators_ = list(map(int,np.linspace(2,100,10)))
learning_rates = np.linspace(0.1,0.9,10)

# Along with the previous hyperparameters, run tests for each combindation

# =============================================================================
# def experiment2():
# =============================================================================
vPrint('Starting experiment 2')
for DataSetName, ds in datasets.items():
    
    ## Get X, Y data for test and train
    Xdata = ds.iloc[:,1:-1]
    Ydata = ds.iloc[:,-1]
    
    ## PRE-PROCESS ALL DATA
    for col in range(Xdata.shape[1]):
        le = preprocessing.LabelEncoder()
# =============================================================================
#         scaler = preprocessing.MinMaxScaler()
# =============================================================================
        Xdata.iloc[:,col] = le.fit_transform(Xdata.iloc[:,col])
    
    # Split data (set random seed)
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(Xdata,
                                                       Ydata,
                                                       test_size=test_size,
                                                       random_state=seed)
    
    dt_boost_score_out_n = []
    dt_boost_score_in_n = []
    dt_boost_time_n = []
        
    ## N estimators variation with fixed, default other values
    for n_estimator in n_estimators_:
        
        model = GradientBoostingClassifier(n_estimators=n_estimator,
                                           random_state = seed)
        
        # Fit the dt to training data
        t = timefit(model,(Xtrain,Ytrain))
        dt_boost_time_n.append(t)
        
        # Score on test data
        dtscore = model.score(Xtest,Ytest)
        
        # Add to scores
        dt_boost_score_out_n.append(dtscore)
        dt_boost_score_in_n.append(model.score(Xtrain,Ytrain))
        
    dt_boost_score_out_lr = []
    dt_boost_score_in_lr = []
    dt_boost_time_lr = []
    
    ## Learning rate variation
    for lr in learning_rates:
        n_estimator = 20
        model = GradientBoostingClassifier(n_estimators=n_estimator,
                                           learning_rate=lr,
                                           random_state = seed)
        
        # Fit the dt to training data
        t = timefit(model,(Xtrain,Ytrain))
        dt_boost_time_lr.append(t)
        
        # Score on test data
        dtscore = model.score(Xtest,Ytest)
        
        # Add to scores
        dt_boost_score_out_lr.append(dtscore)
        dt_boost_score_in_lr.append(model.score(Xtrain,Ytrain))
    
    ## Max depth in learners
    dt_boost_score_out_md = []
    dt_boost_score_in_md = []
    dt_boost_time_md = []
    for md in max_depths:
        n_estimator = 20
        learning_rate = 0.1
        model = GradientBoostingClassifier(n_estimators=n_estimator,
                                           learning_rate=learning_rate,
                                           max_depth = md,
                                           random_state = seed)
        
        # Fit the dt to training data
        t = timefit(model,(Xtrain,Ytrain))
        dt_boost_time_md.append(t)
        
        # Score on test data
        dtscore = model.score(Xtest,Ytest)
        
        # Add to scores
        dt_boost_score_out_md.append(dtscore)
        dt_boost_score_in_md.append(model.score(Xtrain,Ytrain))
    
    ## Min samples
    dt_boost_score_out_ms = []
    dt_boost_score_in_ms = []
    dt_boost_time_ms = []
    for ms in min_samples:
        n_estimators = 20
        learning_rate = 0.1
        max_depth = 5
        model = GradientBoostingClassifier(n_estimators=n_estimator,
                                           learning_rate=learning_rate,
                                           max_depth = max_depth,
                                           min_samples_leaf=ms,
                                           random_state = seed)
        
        # Fit the dt to training data
        t = timefit(model,(Xtrain,Ytrain))
        dt_boost_time_ms.append(t)
        
        # Score on test data
        # Add to scores
        dt_boost_score_out_ms.append(model.score(Xtest,Ytest))
        dt_boost_score_in_ms.append(model.score(Xtrain,Ytrain))
    
    #######################
    # Create plot
    #######################
    fig, ax =  plt.subplots(2,2,figsize=(12,10),dpi=200)
    ## Plot fit times on second axis for each tree
    
    #Boosting n_estimators
    ax00 = ax[0][0].twinx()
    l1 = ax[0][0].plot(n_estimators_,dt_boost_score_in_n,label='Training Score')
    l2 = ax[0][0].plot(n_estimators_,dt_boost_score_out_n,label='Test Score')
    l3 = ax00.plot(n_estimators_,dt_boost_time_n,'k--',label='Training Time')
    ax00.set_ylabel('Training Time (seconds)')
    ls = l1+ l2 + l3
    lb = [l.get_label() for l in ls]
    ax[0][0].set_xlabel('n_estimators')
    ax[0][0].set_ylabel('Score')
    ax[0][0].grid()
    ax[0][0].legend(ls,lb,loc=0)
    
    #Boosting learning rate
    ax01 = ax[0][1].twinx()
    l1 = ax[0][1].plot(learning_rates,dt_boost_score_in_lr,label='Training Score')
    l2 = ax[0][1].plot(learning_rates,dt_boost_score_out_lr,label='Test Score')
    l3 = ax00.plot(learning_rates,dt_boost_time_lr,'k--',label='Training Time')
    ax00.set_ylabel('Training Time (seconds)')
    ls = l1+ l2 + l3
    lb = [l.get_label() for l in ls]
    ax[0][1].set_xlabel(f'learning_rate (n_estimators = {n_estimator}')
    ax[0][1].set_ylabel('Score')
    ax[0][1].grid()
    ax[0][1].legend(ls,lb,loc=0)
    
    #Boosting max_depth
    ax10 = ax[1][0].twinx()
    l1 = ax[1][0].plot(max_depths,dt_boost_score_in_md,label='Training Score')
    l2 = ax[1][0].plot(max_depths,dt_boost_score_out_md,label='Test Score')
    l3 = ax10.plot(max_depths,dt_boost_time_md,'k--',label='Training Time')
    ax10.set_ylabel('Training Time (seconds)')
    ls = l1+ l2 + l3
    lb = [l.get_label() for l in ls]
    ax[1][0].set_xlabel('Max_Depth')
    ax[1][0].set_ylabel('Score')
    ax[1][0].grid()
    ax[1][0].legend(ls,lb,loc=0)
    
    #Boosting min_samples_leaf
    ax11 = ax[1][1].twinx()
    l1 = ax[1][1].plot(min_samples,dt_boost_score_in_ms,label='Training Score')
    l2 = ax[1][1].plot(min_samples,dt_boost_score_out_ms,label='Test Score')
    l3 = ax11.plot(min_samples,dt_boost_time_ms,'k--',label='Training Time')
    ax10.set_ylabel('Training Time (seconds)')
    ls = l1+ l2 + l3
    lb = [l.get_label() for l in ls]
    ax[1][1].set_xlabel('Min_samples_leaf')
    ax[1][1].set_ylabel('Score')
    ax[1][1].grid()
    ax[1][1].legend(ls,lb,loc=0)
    

###############################################################################
# Setup GridSeachCV for Decision Tree with Boosting for each dataset
###############################################################################
    if run_grid_search:    
        estimator = GradientBoostingClassifier(random_state = seed) 
        vPrint(f'Starting parameter grid search for {estimator}: {DataSetName}\n')
    
        param_grid = {
            'n_estimators':n_estimators_,
            'learning_rate':learning_rates,
            'max_depth':max_depths,
            'min_samples_leaf':min_samples,
            }
        
        fold = []
        for i in range(Xdata.shape[0]):
            if Xdata.index[i] in Xtrain.index:
                fold.append(-1)
            else:
                fold.append(0)
                
        ps = model_selection.PredefinedSplit(fold) # Fix the fold on the train data
        
        clf = model_selection.GridSearchCV(estimator=estimator, 
                                           param_grid = param_grid, 
                                           verbose=3 if verbose else 0,
                                           scoring = 'accuracy',
                                           n_jobs = n_jobs,
                                           return_train_score = True,
                                           cv=ps,
                                           )
    
        clf.fit(Xdata,Ydata)
    
        vPrint(f'GridSearchCV Complete for {DataSetName} using {estimator}.')
        
        best_estimator_ = clf.best_estimator_
        best_estimator_.fit(Xtrain,Ytrain)

        # Add to plot
        ax[0][0].plot(best_estimator_.n_estimators,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[0][1].plot(best_estimator_.learning_rate,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[1][0].plot(best_estimator_.max_depth,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[1][1].plot(best_estimator_.min_samples_leaf,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        
        
    
    fig.tight_layout()
    plt.suptitle(f'Boosted Tree - {DataSetName}\n')
    if savefig:
        plt.savefig(f'Images/BoostedTreeClassifier_{DataSetName}_Figure.png')
    plt.show() # Save fig
###############################################################################
# Neural Network - Using Multi-Layer Perceptron MLP Classifier
###############################################################################

# =============================================================================
# def experiment3():
# =============================================================================
vPrint('Starting experiment 3')
max_iter = 1000
n_iter_no_change = 20
for DataSetName, ds in datasets.items():
    
    ## Get X, Y data for test and train
    Xdata = ds.iloc[:,1:-1]
    Ydata = ds.iloc[:,-1]
    
    ## PRE-PROCESS ALL DATA
    for col in range(Xdata.shape[1]):
        le = preprocessing.LabelEncoder()
        scaler = preprocessing.MinMaxScaler()
        Xdata.iloc[:,col] = le.fit_transform(Xdata.iloc[:,col])
        Xdata.iloc[:,col] = scaler.fit_transform(np.array(Xdata.iloc[:,col]).reshape(-1,1))
        
    # Split data (set random seed)
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(Xdata,
                                                       Ydata,
                                                       test_size=test_size,
                                                       random_state=seed)
    
    ##Vary Learning_rate using learning_rate_init
    lr_rates = np.linspace(0.0005,0.001,10)
    
    mlp_out_score_lr = []
    mlp_in_score_lr = []
    mlp_time_lr = []
    
    # Start figure before loop to add loss_curves
    fig, ax =  plt.subplots(2,2,figsize=(12,10),dpi=200)
    # Use constant rate
    for lr_ in lr_rates:
        
        mlp_loss_lr = []
        model = MLPClassifier(learning_rate_init=lr_,
                              early_stopping = True,
                              hidden_layer_sizes = (100,),
                              max_iter = max_iter,
                              n_iter_no_change = n_iter_no_change,
                              random_state = seed)
    
        t = timefit(model,(Xtrain,Ytrain))
        
        mlp_time_lr.append(t)
        
        mlp_out_score_lr.append(model.score(Xtest,Ytest))
        mlp_in_score_lr.append(model.score(Xtrain,Ytrain))
        
        ax[0][0].plot(model.loss_curve_,label=f'LR={round(lr_,5)}')
        ax[0][1].plot(model.validation_scores_,label=f'LR={round(lr_,5)}')
        
    # Vary the # of nodes in each hidden layer 
    n_h = list(map(int,np.linspace(10, 200,10)))
    # Set constant learning_rate == 0.0005 with early stopping and random seed
    for n_h_ in n_h:
        
        mlp_loss_nh = []
        
        model = MLPClassifier(learning_rate_init=0.0005,
                              early_stopping = True,
                              hidden_layer_sizes = (n_h_,),
                              max_iter = max_iter,
                              n_iter_no_change = n_iter_no_change,
                              random_state = seed)
        
        t = timefit(model,(Xtrain,Ytrain))
        
        ax[1][0].plot(model.loss_curve_,label=f'n_h={n_h_}')
        ax[1][1].plot(model.validation_scores_,label=f'n_h={n_h_}')
        
        
    
    #######################
    # Create plot
    #######################
    # Loss for lr_
    ax[0][0].set_xlabel('Iterations')
    ax[0][0].set_ylabel('Loss')
    ax[0][0].grid()
    # Validation for lr_
    ax[0][1].set_xlabel('Iterations')
    ax[0][1].set_ylabel('Validation Score')
    ax[0][1].grid()
    # Loss for n_h_
    ax[1][0].set_xlabel('Iterations')
    ax[1][0].set_ylabel('Loss')
    ax[1][0].grid()
    # Validation for lr_
    ax[1][1].set_xlabel('Iterations')
    ax[1][1].set_ylabel('Validation Score')
    ax[1][1].grid()
    
###############################################################################
# Setup GridSeachCV for MLP
###############################################################################
    if run_grid_search:
        estimator = MLPClassifier(max_iter = max_iter,
                                  random_state = seed,
                                  early_stopping = True,
                                  )
        
        vPrint(f'Starting parameter grid search for {estimator}: {DataSetName}\n')
        
        param_grid = {
            'learning_rate_init':lr_rates,
            'activation':['identity', 'logistic', 'tanh', 'relu'],
            'hidden_layer_sizes':n_h,
            }
        
        fold = []
        for i in range(Xdata.shape[0]):
            if Xdata.index[i] in Xtrain.index:
                fold.append(-1)
            else:
                fold.append(0)
                
        ps = model_selection.PredefinedSplit(fold) # Fix the fold on the train data
        
        clf = model_selection.GridSearchCV(estimator=estimator, 
                                           param_grid = param_grid, 
                                           verbose=3 if verbose else 0,
                                           scoring = 'accuracy',
                                           n_jobs = n_jobs,
                                           return_train_score = True,
                                           cv=ps,
                                           )
    
        clf.fit(Xdata,Ydata)
    
        best_estimator_ = clf.best_estimator_
        best_estimator_.fit(Xtrain,Ytrain)

        vPrint(f'GridSearchCV Complete for {DataSetName} using {estimator}.')
        
        ### ADD TO PLOT ###
        # Add to loss curve plots
        ax[0][0].plot(best_estimator_.loss_curve_,'k-',label='Best Estimator')
        ax[1][0].plot(best_estimator_.loss_curve_,'k-',label='Best Estimator')
        ax[0][0].legend()
        ax[1][0].legend()
        # Add to validation curve plots
        ax[0][1].plot(best_estimator_.validation_scores_,'k-',label='Best Estimator')
        ax[1][1].plot(best_estimator_.validation_scores_,'k-',label='Best Estimator')
        ax[0][1].legend()
        ax[1][1].legend()

    fig.tight_layout()
    plt.suptitle(f'Multi-Layer Perceptron Classifier - {DataSetName}\n')
    if savefig:
        plt.savefig(f'Images/Multi-Layer_Perceptron_Classifier_{DataSetName}_Figure.png')
    plt.show() # Save fig
    
###############################################################################
# SVM - Support Vector Machines
###############################################################################

# =============================================================================
# def experiment4():
# =============================================================================
vPrint('Starting experiment 4')
for DataSetName, ds in datasets.items():
    
    ## Get X, Y data for test and train
    Xdata = ds.iloc[:,1:-1]
    Ydata = ds.iloc[:,-1]
    
    ## PRE-PROCESS ALL DATA
    for col in range(Xdata.shape[1]):
        le = preprocessing.LabelEncoder()
        scaler = preprocessing.MinMaxScaler()
        Xdata.iloc[:,col] = le.fit_transform(Xdata.iloc[:,col])
        Xdata.iloc[:,col] = scaler.fit_transform(np.array(Xdata.iloc[:,col]).reshape(-1,1))
    
    # Split data (set random seed)
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(Xdata,
                                                       Ydata,
                                                       test_size=test_size,
                                                       random_state=seed)
    
    loss_ = ['hinge', 'modified_huber', 'squared_hinge', 'perceptron',
             'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
    alpha_ = np.linspace(0.0001,0.001,20)
    
    # Start Plot
    fig, ax =  plt.subplots(1,2,figsize=(12,10),dpi=200)
    # 
    svm_time_ls = []
    svm_out_score_ls = []
    svm_in_score_ls = []
    
    for loss in loss_:
        model = SGDClassifier(loss = loss,
                                  learning_rate = 'optimal',
                                  early_stopping = True,
                                  random_state = seed)
    
    
        vPrint(f'LinearSVC: {model} fitting...')
        svm_time_ls.append(timefit(model,args=(Xtrain,Ytrain)))
        vPrint(f'SGDClassifier: {model} fitting...')
        
        timefit(model,args=(Xtrain,Ytrain))
        
        svm_out_score_ls.append(timescore(model,args=(Xtest,Ytest))[1])
        svm_in_score_ls.append(timescore(model,args=(Xtrain,Ytrain))[1])
        
    svm_time_a = []
    svm_out_score_a = []
    svm_in_score_a = []
    
    for alpha in alpha_:
        model = SGDClassifier(loss = 'hinge',
                                  alpha = alpha,
                                  learning_rate = 'optimal',
                                  early_stopping = True,
                                  random_state = seed)


        vPrint(f'LinearSVC: {model} fitting...')
        svm_time_a.append(timefit(model,args=(Xtrain,Ytrain)))
        vPrint(f'SGDClassifier: {model} fitting...')
        
        timefit(model,args=(Xtrain,Ytrain))
        
        svm_out_score_a.append(model.score(Xtest,Ytest))
        svm_in_score_a.append(model.score(Xtrain,Ytrain))
        
    #######################
    # Create plot
    #######################
    # Loss for lr_
    ax[0].plot(loss_,svm_out_score_ls,'ro',label='Test Score')
    ax[0].plot(loss_,svm_in_score_ls,'bo',label='Training Score')
    
    ax[0].set_xticklabels(loss_,rotation=90)
    ax[0].set_xlabel('Loss Function')
    ax[0].set_ylabel('Score')
    ax[0].grid()
    ax[0].legend()
    # Validation for alpha_
    ax1 = ax[1].twinx()
    ax[1].plot(alpha_,svm_out_score_a,label='Test Score')
    ax[1].plot(alpha_,svm_in_score_a,label='Training Score')
    ax[1].set_xticks(alpha_)
    ax[1].set_xticklabels(alpha_.round(5),rotation=45)
    ax[1].set_xlabel('Alpha')
    ax[1].set_ylabel('Score')
    ax[1].grid()
    ax[1].legend()

###############################################################################
# Setup GridSeachCV for MLP
###############################################################################
    if run_grid_search:
        estimator =SGDClassifier(learning_rate = 'optimal',
                                 early_stopping = True,
                                 random_state = seed)
        
        vPrint(f'Starting parameter grid search for {estimator}: {DataSetName}\n')
        
        param_grid = {
            'alpha':alpha_,
            'loss':loss_,
            }
        
        fold = []
        for i in range(Xdata.shape[0]):
            if Xdata.index[i] in Xtrain.index:
                fold.append(-1)
            else:
                fold.append(0)
                
        ps = model_selection.PredefinedSplit(fold) # Fix the fold on the train data
        
        clf = model_selection.GridSearchCV(estimator=estimator, 
                                           param_grid = param_grid, 
                                           verbose=3 if verbose else 0,
                                           scoring = 'accuracy',
                                           n_jobs = n_jobs,
                                           return_train_score = True,
                                           cv=ps,
                                           )
    
        clf.fit(Xdata,Ydata)
    
        vPrint(f'GridSearchCV Complete for {DataSetName} using {estimator}.')
        
        best_estimator_ = clf.best_estimator_
        best_estimator_.fit(Xtrain,Ytrain)

        ### ADD TO PLOT ###
        # Add to loss curve plots
        ax[0].plot(best_estimator_.loss,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[1].plot(best_estimator_.alpha,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[0].legend()
        ax[1].legend()
        
    fig.tight_layout()
    plt.suptitle(f'SGDClassifier - {DataSetName}\n')
    if savefig:
        plt.savefig(f'Images/SGD_Classifier_{DataSetName}_Figure.png')
    plt.show() # Save fig


###############################################################################
# KNN - K-Nearest Neighbors
###############################################################################

# =============================================================================
# def experiment5():
# =============================================================================
ks = list(map(int,np.linspace(1,20,10)))
w_ = ['uniform','distance']
algo_ = ['ball_tree','kd_tree','brute']
p_ = list(map(int,np.linspace(1,5,5)))

vPrint('Starting experiment 5')
for DataSetName, ds in datasets.items():
    
    ## Get X, Y data for test and train
    Xdata = ds.iloc[:,1:-1]
    Ydata = ds.iloc[:,-1]
    
    ## PRE-PROCESS ALL DATA
    for col in range(Xdata.shape[1]):
        le = preprocessing.LabelEncoder()
        scaler = preprocessing.MinMaxScaler()
        Xdata.iloc[:,col] = le.fit_transform(Xdata.iloc[:,col])
        Xdata.iloc[:,col] = scaler.fit_transform(np.array(Xdata.iloc[:,col]).reshape(-1,1))
    
    # Split data (set random seed)
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(Xdata,
                                                       Ydata,
                                                       test_size=test_size,
                                                       random_state=seed)
    
    knn_score_in_ = []
    knn_score_out_ = []
    query_time_out = []
    query_time_in = []
    for k in ks:
        
        model = KNeighborsClassifier(n_neighbors=k,
                                     n_jobs=n_jobs, # Specify using all cpus for query
                                     )
        timefit(model,args=(Xtrain,Ytrain))
        start = time()
        knn_score_in_.append(timescore(model,args=(Xtrain,Ytrain))[1])
        end = time()
        knn_score_out_.append(timescore(model,args=(Xtest,Ytest))[1])
        end2 = time()
        
    knn_score_in_p = []
    knn_score_out_p = []
    
    for p in p_:
        
        model = KNeighborsClassifier(n_neighbors=10,
                                     n_jobs=n_jobs, # Specify using all cpus for query
                                     p = p
                                     )
        
        timefit(model,args=(Xtrain,Ytrain))
        start = time()
        knn_score_in_p.append(timescore(model,args=(Xtrain,Ytrain))[1])
        end = time()
        knn_score_out_p.append(timescore(model,args=(Xtest,Ytest))[1])
        end2 = time()
        
    
    #######################
    # Create plot
    #######################
        
    fig, ax =  plt.subplots(1,2,figsize=(12,10),dpi=200)
    ax[0].plot(ks,knn_score_in_,label='Training Score')
    ax[0].plot(ks,knn_score_out_,label='Test Score')
    ax[0].set_xlabel('Neighbors')
    ax[0].set_ylabel('Score')
    ax[0].grid()
    
    ax[1].plot(p_,knn_score_in_p,label='Training Score')
    ax[1].plot(p_,knn_score_out_p,label='Test Score')
    ax[1].set_xlabel('Power Parameter (p)')
    ax[1].set_ylabel('Score')
    ax[1].grid()
    
    
    
###############################################################################
# Setup GridSeachCV for KNN
###############################################################################
    if run_grid_search:
        estimator = KNeighborsClassifier(
                                 n_jobs = n_jobs,
                                 )
        
        vPrint(f'Starting parameter grid search for {estimator}: {DataSetName}\n')
        
        param_grid = {
            'n_neighbors':ks,
            'weights':w_,
            'algorithm':algo_,
            'p':p_,
            }
        
        fold = []
        for i in range(Xdata.shape[0]):
            if Xdata.index[i] in Xtrain.index:
                fold.append(-1)
            else:
                fold.append(1)
        
        ps = model_selection.PredefinedSplit(fold) # Fix the fold on the train data
        
        clf = model_selection.GridSearchCV(estimator=estimator, 
                                           param_grid = param_grid, 
                                           verbose=3 if verbose else 0,
                                           scoring = 'accuracy',
                                           n_jobs = n_jobs,
                                           return_train_score = True,
                                           cv=ps,
                                           )
    
        clf.fit(Xdata,Ydata)
        
        vPrint(f'GridSearchCV Complete for {DataSetName} using {estimator}.')

        best_estimator_ = clf.best_estimator_
        best_estimator_.fit(Xtrain,Ytrain)

        # Add to plot
        ax[0].plot(best_estimator_.n_neighbors,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[0].legend()
        ax[1].plot(best_estimator_.p,best_estimator_.score(Xtest,Ytest),'k*',label='Best Estimator')
        ax[1].legend()

    fig.tight_layout()
    plt.suptitle(f'KNN - {DataSetName}\n')
    if savefig:
        plt.savefig(f'Images/KNN_{DataSetName}_Figure.png')
    plt.show() # Save fig
    

