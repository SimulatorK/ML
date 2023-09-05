## CS7641 Machine Learning - Assignment 1
"""
    Create models for Decision Tree,DT with boosting, SVM, KNN, Neural Nets 





"""
# Simple Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from  sklearn import model_selection
# Decision Tree with boosting
from sklearn.ensemble import GradientBoostingClassifier
# Neural Network MLP
from sklearn.neural_network import MLPClassifier
# KNN
from sklearn.neighbors import KNeighborsClassifier
# Support Vector Machine - SVM
from sklearn.svm import LinearSVC

import os
import pandas as pd
#simport random
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Load data sets
seed = 1

if __name__ == '__main__':
    os.chdir(os.path.split(__file__)[0])
    
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
    'Room Occupancy':ds2,
    'Mushroom Classification':ds3,
    'Student Dropout':ds4,
    }

#Train Test Split for all experiments 
test_size = 0.1 

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

# Min samples to try
min_samples = list(map(int,np.linspace(200,2,10)))
# Max depth to try
max_depths = list(map(int,np.linspace(1,20,10)))


#### Run test for each data set
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
        
    #######################
    # Create plot
    #######################
    fig, ax =  plt.subplots(2,2,figsize=(12,10),dpi=200)
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
    
    fig.tight_layout()
    plt.suptitle(f'Decision Tree - {DataSetName}\n')
    plt.savefig(f'Images/DecisionTreeClassifier_{DataSetName}_Figure.png')
    plt.show() # Save fig

    
###############################################################################
# Decision Tree with boosting using GBC
###############################################################################
n_estimators_ = list(map(int,np.linspace(2,100,10)))
learning_rates = np.linspace(0.1,0.9,10)

# Along with the previous hyperparameters, run tests for each combindation


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
    
    fig.tight_layout()
    plt.suptitle(f'Boosted Tree - {DataSetName}\n')
    plt.savefig(f'Images/BoostedTreeClassifier_{DataSetName}_Figure.png')
    plt.show() # Save fig
    
    # 


###############################################################################
# Neural Network - Using Multi-Layer Perceptron MLP Classifier
###############################################################################


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
                              random_state = seed)
    
        t = timefit(model,(Xtrain,Ytrain))
        
        mlp_time_lr.append(t)
        
        mlp_out_score_lr.append(model.score(Xtest,Ytest))
        mlp_in_score_lr.append(model.score(Xtrain,Ytrain))
        
        ax[0][0].loglog(model.loss_curve_,label=f'LR={round(lr_,5)}')
    
    #######################
    # Create plot
    #######################
    
    ax[0][0].set_xlabel('Iterations')
    ax[0][0].set_ylabel('Loss')
    ax[0][0].grid()
    ax[0][0].legend()
    
    ax[1][0].plot(lr_rates,mlp_time_lr,)
    ax[1][0].set_xlabel('Learning Rate')
    ax[1][0].set_ylabel('Training Time (sec)')
    ax[1][0].grid()
    
    
    # Vary on 
    

    fig.tight_layout()
    plt.suptitle(f'Multi-Layer Perceptron Classifier - {DataSetName}\n')
    plt.savefig(f'Images/Multi-Layer_Perceptron_Classifier_{DataSetName}_Figure.png')
    plt.show() # Save fig
















