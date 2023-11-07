## CS7641 Machine Learning - Assignment 3

from sklearn import preprocessing
from  sklearn import model_selection
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.metrics import silhouette_samples, silhouette_score
# Neural Network MLP
from sklearn.neural_network import MLPClassifier
from collections import Counter
import os
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

from scipy.spatial.distance import cdist
from sklearn import mixture
import math
from matplotlib.patches import Ellipse

verbose = True
###############################################################################
# Helper Functions
###############################################################################

def vPrint(text: str = '', verbose: bool = verbose):
    if verbose:
        print(text)

    
# Load data sets
seed = 903860493

if __name__ == '__main__':
    os.chdir(os.path.split(__file__)[0])
 


































