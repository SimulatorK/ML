
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import pipeline

class bcolors:
    ENDC = '\033[m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    
def colorit(text: str = '',color: str = '', verbose: bool = True):
    if verbose:
        print(color + text + bcolors.ENDC)
        
        
class experiment:
    
    def __init__(self,
                 model,
                 args,
                 kwargs,
                 data,
                 test_size,
                 seed: int = None,
                 ):
        
        self.model = model(*args,**kwargs)
        self.data = data
        self.test_size = test_size
        self.seed = random.randint(1,1e6) if not seed else seed
    
    
    def setup(self):
        self.Xdata = self.data[:,:-1]
        self.Ydata = self.data[:,-1]
        
        self.pipe = pipeline.Pipeline([('label',preprocessing.LabelEncoder()),
                                       ('scaler',preprocessing.MinMaxScaler()),
                                        ])
        
        Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(self.Xdata,
                                                                        self.Ydata,
                                                                        test_size=self.test_size,
                                                                        random_state=self.seed)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        