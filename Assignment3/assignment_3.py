## CS7641 Machine Learning - Assignment 3

from sklearn import preprocessing
from  sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# Neural Network MLP
from sklearn.neural_network import MLPClassifier

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
 
## Select Data Sets
creditDefaultFile = r"data/default of credit card clients.xls"
mushroomClassificationFile = r"data/secondary+mushroom+dataset/MushroomDataset/secondary_data.csv"

creditDefaultFile = os.path.join(os.getcwd(),creditDefaultFile)

ds1 = pd.read_excel(creditDefaultFile,skiprows=1)
ds3 = pd.read_csv(mushroomClassificationFile)

datasets = {
    'Credit Default':ds1,
    'Mushroom Classification':ds3
    }

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


    

