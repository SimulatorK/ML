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

def plot_results(X, Y_, means, covariances, index, title):
    
    color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

    splot = plt.subplot()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = math.atan2(u[1], u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9.0, 5.0)
    plt.ylim(-3.0, 6.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()
    
# Load data sets
seed = 903860493

if __name__ == '__main__':
    os.chdir(os.path.split(__file__)[0])
 
## Select Data Sets
mushroomClassificationFile = r"data/secondary+mushroom+dataset/MushroomDataset/secondary_data.csv"
prostateClassificationFile = r"data/mnist/dataset"
beanClassificationFile = r"data/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"

mushroomClassificationFile = os.path.join(os.getcwd(),mushroomClassificationFile) # fewer features, 2 classes
#prostateClassificationFile = os.path.join(os.getcwd(),prostateClassificationFile) # lots of features, 2 classes
beanClassificationFile = os.path.join(os.getcwd(),beanClassificationFile)

ds1 = pd.read_csv(mushroomClassificationFile)
#ds2 = pd.read_csv(prostateClassificationFile,skiprows=108,nrows=20000,header=None)
ds3 = pd.read_excel(beanClassificationFile)

datasets = {
    'Mushroom Classification':ds1,
    #'Prostate Cancer Classification':ds2,
    'Bean Classification':ds3,
    }

#Train Test Split for all experiments 
test_size = 0.1

# Verbosity level
run_grid_search = True

# Run n_jobs in parallel
n_jobs = -14

# Savfigure flag
savefig = True

cores = min(n_jobs,os.cpu_count()) if n_jobs > 0 else max(1,os.cpu_count() + n_jobs) 
cores_ = f'{cores} cores' if cores > 1 else '1 core'
vPrint(f'Using {cores_} to process')

#################################################################################

## Run analysis on both experiments

for dataSetName, dataset in datasets.items():
    
    vPrint(f'Starting analysis for: {dataSetName}')
    
    
###############################################################################
# DIMENSION REDUCTION
###############################################################################

    # Split data into test and train
    Xdata = dataset.iloc[:,:-1] # X data is all columns except first
    Ydata = dataset.iloc[:,-1] # Y data is the last column
    
    # Plot Xdata distribution and determine statistics before transforming the data
    # Encode labels in data
    for col in Xdata:
        try:
            list(map(float,Xdata[col]))
        except:
            le = preprocessing.LabelEncoder()
            Xdata[col] = le.fit_transform(Xdata[col])
            
        if sum(np.isnan(Xdata[col])) or sum(np.isinf(Xdata[col])):
            # set isnan or ising columns to 0
            Xdata.drop(col)
    
    # Encode Ydata
    le = preprocessing.LabelEncoder()
    Ydata = le.fit_transform(Ydata)
    
    n_classes = len(Counter(dataset[dataset.columns[-1]]).keys())
    n_features = dataset.shape[1] - 1
    features = list(dataset.columns)
    
    ## Statistical Plots
    max_c = 4
    nr = n_features // (max_c - 1) - 1
    nc = n_features // nr
    
    fig, ax = plt.subplots(nr,nc,figsize=(8,6),dpi=200)
    
    xk = []
    for col in Xdata:
        coli = list(Xdata.columns).index(col)
        c = coli % nc
        
        r = coli // nc
        
        print(coli,r,c)
        sns.distplot(Xdata[col],kde=False,fit=ss.norm,ax=ax[r][c])
        
        k = ss.kurtosis(Xdata[col])
        print(f'col: Kurtosis = {k}')
        xk.append(k)
        
    fig.tight_layout()
    #plt.savefig(f'Images/{dataSetName}_distributions.png')
    plt.show()
    
    
    fig, ax = plt.subplots(figsize = (8,6),dpi = 200)
    ax.plot(range(0,n_features),xk)
    ax.set_xlabel('Component')
    ax.set_ylabel('Kurtosis')
    ax.grid()
    plt.title(f'Kurtosis_{dataSetName}')
    fig.tight_layout()
    #plt.savefig(f'Images/Kurtosis_{dataSetName}.png')
    plt.show()
    
    Xdata = np.array(Xdata)
    
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(Xdata,
                                                       Ydata,
                                                       test_size=test_size,
                                                       random_state=seed)
    
    ## Preprocess training data and apply to test data
    scaler = preprocessing.MinMaxScaler()   
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    
    n_features_ = range(2,n_features)
    
    ###########################################################################
    # ANALYSIS
    ###########################################################################    
    
    
    vPrint('Dimensionality Reduction Algorithms')
    vPrint(f'\tPCA:')
    
    pca_mse = []
    pca_k = []
    for n_f in n_features_:
            
        # Transform training data
        pca = PCA(n_components = n_f, random_state=seed)
        xpca = pca.fit_transform(Xtrain)
        xpca_i = pca.inverse_transform(xpca)
        pca_mse.append(np.square(Xtrain-xpca_i).mean())
    
    ## Plot eigenvalues of principal components
    cum_ev = [sum(pca.explained_variance_[:i]) for i in range(len(pca.explained_variance_))]
    cum_ev = cum_ev/max(cum_ev)
    
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    l1 = ax.plot(pca.explained_variance_,label='PCA Exlained Variance')
    l2 = ax.plot(cum_ev,label='Normalized Cumulative PCA Exlained Variance')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Eigenvalue (Explained Variance)')
    ax.grid()
    
    # Plot the MSE of the transformed data for different n_components
    ax2 = ax.twinx()
    l3 = ax2.plot(n_features_,pca_mse,'k-',label = 'MSE: PCA_inverse vs. Data')
    ls = l1+ l2 + l3
    lb = [l.get_label() for l in ls]
    ax2.set_ylabel('MSE - Transformed vs. Data')
    
    plt.legend(ls,lb,loc=0)
    plt.title(f'PCA_{dataSetName}_with_MSE')
    plt.savefig(f'Images/PCA_{dataSetName}_with_MSE.png')
    plt.show()
    
    # Plot first and second PCA with cluster color
    fig, ax = plt.subplots(figsize = (8,6), dpi = 200)
    ax.scatter(xpca[:,0],xpca[:,1],c=Ytrain)
    ax.set_xlabel('PCA 0')
    ax.set_ylabel('PCA 1')
    ax.grid()
    fig.tight_layout()
    plt.title(f'PCA0_vs_PCA1_{dataSetName}')
    plt.savefig(f'Images/PCA0_vs_PCA1_{dataSetName}.png')
    plt.show()
    
    vPrint(f'\tICA:')
    # Transform training data
    fig, ax = plt.subplots(n_features,1,figsize = (8,6),dpi = 200)
    ica_mse = []
    ica_k_mean = []
    ica_std = []
    for n_f in n_features_:
        ica = FastICA(n_components = n_f, random_state=seed)
        xica = ica.fit_transform(Xtrain)
        xica_i = ica.inverse_transform(xica)
        ica_mse.append(np.square(Xtrain-xica_i).mean())
        ica_k = ss.kurtosis(xica)
        ica_std.append(np.std(xica))
        ica_k_mean.append(ica_k.mean())
        ax[n_f].plot(range(n_f),ica_k,label=f'n = {n_f}')
        ax[n_f].grid()
        
        ax[n_f].set_xlabel('# of Components')
    plt.ylabel("Kurtosis")
    fig.tight_layout()
    plt.title(f'ICA_{dataSetName}_Kurtosis')
    #plt.savefig(f'Images/ICA_{dataSetName}_Kurtosis.png')
    plt.show()
    
    fig, ax = plt.subplots(figsize = (8,6), dpi = 200)
    ax.plot(n_features_,ica_k_mean)
    ax.grid()
    ax.set_xlabel('# of Components')
    ax.set_ylabel('Average Transformed kurtosis')
    plt.title(f'ICA_AverageKurtosis_{dataSetName}')
    fig.tight_layout()
    plt.savefig(f'Images/ICA_AverageKurtosis_{dataSetName}.png')
    plt.show()
    
    ## Plot 
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    ax.plot(ica.mean_)
    ax.set_xlabel('Independent Component')
    ax.set_ylabel('ICA Mean')
    ax.grid()
    plt.title(f'ICA_{dataSetName}_means')
    #plt.savefig(f'Images/ICA_{dataSetName}_means.png')
    plt.show()
    
    ## Plot mse of principal components
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    ax.plot(n_features_,ica_mse)
    ax.set_xlabel('Independent Component')
    ax.set_ylabel('MSE: ICA_inverse vs. Data')
    ax.grid()
    plt.title(f'ICA_{dataSetName}_MSE')
    #plt.savefig(f'Images/ICA_{dataSetName}_MSE.png')
    plt.show()

    ## Plot xica
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    ax.plot(xica)
    ax.set_xlabel('Independent Component')
    ax.set_ylabel('')
    ax.grid()
    plt.title(f'ICA_{dataSetName}')
    #plt.savefig(f'Images/ICA_{dataSetName}.png')
    plt.show()
    
    # Plot first and second PCA with cluster color
    fig, ax = plt.subplots(figsize = (8,6), dpi = 200)
    ax.scatter(xica[:,0],xica[:,1],c=Ytrain)
    ax.set_xlabel('ICA 0')
    ax.set_ylabel('ICA 1')
    ax.grid()
    fig.tight_layout()
    plt.title(f'ICA0_vs_ICA1_{dataSetName}')
    plt.savefig(f'Images/ICA0_vs_ICA1_{dataSetName}.png')
    plt.show()
    
    vPrint(f'\tGRP:')
    # Transform training data
    grp_mse = []
    grp_k_mean = []
    grp_n_features = range(2,200)
    for n_f in grp_n_features:
        grp = GaussianRandomProjection(n_components=n_f,random_state=seed,eps = 0.1)
        xgrp = grp.fit_transform(Xtrain)
        xgrp_i = xgrp.dot(grp.components_) + np.mean(Xtrain,axis=0)
        grp_mse.append(np.square(Xtrain-xgrp_i).mean())
        grp_k_mean.append(ss.kurtosis(xgrp).mean())

    grp_k_v_mse = np.array(grp_k_mean) / np.array(grp_mse)
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    ax.plot(grp_n_features,grp_mse)
    ax.grid()
    ax.set_xlabel('# of Components')
    ax.set_ylabel('Reconstruction Error')
    plt.title(f'GRP_{dataSetName}_ReconstructionError')
    fig.tight_layout()
    plt.savefig(f'Images/GRP_{dataSetName}_ReconstructionError.png')
    plt.show()

    # Plot first and second GRP with cluster color
    fig, ax = plt.subplots(figsize = (8,6), dpi = 200)
    ax.scatter(xgrp[:,0],xgrp[:,1],c=Ytrain)
    ax.set_xlabel('GRP 0')
    ax.set_ylabel('GRP 1')
    ax.grid()
    fig.tight_layout()
    plt.title(f'GRP0_vs_GRP1_{dataSetName}')
    plt.savefig(f'Images/GRP0_vs_GRP1_{dataSetName}.png')
    plt.show()
    
    # t-SNE
    vPrint(f'\tt-SNE:')
    
    # Transform training data
    tsne_mse = []
    p = 2
    tsne = TSNE(2,random_state=seed,verbose=verbose,n_jobs=n_jobs,perplexity=p)
    xtsne = tsne.fit_transform(Xtrain)
# =============================================================================
#         xtsne_i = np.dot(xtsne,tsne.components_)
#         grp_mse.append(np.square(Xtrain-xgrp_i).mean())
# =============================================================================
    # Plot first and second PCA with cluster color
    fig, ax = plt.subplots(figsize = (8,6), dpi = 200)
    ax.scatter(xtsne[:,0],xtsne[:,1],c=Ytrain)
    ax.set_xlabel('t-SNE 0')
    ax.set_ylabel('t-SNE 1')
    ax.grid()
    fig.tight_layout()
    plt.title(f'tSNE0_vs_tSNE1{dataSetName}')
    plt.savefig(f'Images/tSNE0_vs_tSNE1{dataSetName}.png')
    plt.show()
    
    
###############################################################################
# CLUSTERING
###############################################################################
    
    def plot_kmeans(kmeans, X, n_clusters, rseed=seed, ax=None):
        labels = kmeans.fit_predict(X)
    
        # plot the input data
        ax = ax or plt.gca()
        ax.axis('equal')
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
        ax.set_xlabel('Training Feature 1')
        ax.set_ylabel('Training Feature 2')
        # plot the representation of the KMeans model
        centers = kmeans.cluster_centers_
        radii = [cdist(X[labels == i], [center]).max()
                 for i, center in enumerate(centers)]
# =============================================================================
#         for c, r in zip(centers, radii):
#             ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
#             
# =============================================================================
    vPrint(f'\tKmeans:')
    # Vary the number of clusters
    silhouette_scores = []
    n_clusters_ = range(2,n_classes*5+1)
    for n_clusters in n_clusters_:
        
        kmeans = KMeans(n_clusters = n_clusters, random_state=seed)
        xkmeans = kmeans.fit_predict(Xtrain)
    
# =============================================================================
#         fig, ax = plt.subplots(figsize=(8,6),dpi=200)
# =============================================================================
# =============================================================================
#         for col in range(Xtrain.shape[1]):
#             ax.scatter(Xtrain[:,col],Ytrain,c=kmeans.labels_)
#             ax.grid()
#         ax.set_ylabel('True Class')
#         ax.set_xlabel('Feature')
#         
#         plt.title(f'KMeans_{dataSetName}_{n_clusters}_clusters')
#         plt.savefig(f'Images/KMeans_{dataSetName}_{n_clusters}_clusters.png')
#         plt.show()
#         
# =============================================================================
        # Get silhouette score
        silhouette_scores.append(silhouette_score(Xtrain,kmeans.labels_))
        
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    ax.plot(n_clusters_,silhouette_scores)
    ax.set_xlabel('# of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.grid()
    plt.title(f'KMeans_{dataSetName}_SilhouetteScore')
    plt.savefig(f'Images/KMeans_{dataSetName}_SilhouetteScore.png')
    plt.show()
    
    # Plot all clusters
# =============================================================================
#     u_l = np.unique(xkmeans)
#     centroids = kmeans.cluster_centers_
# =============================================================================
    kmeans_n_clusters = n_clusters_[np.argmax(silhouette_scores)]
    kmeans_ = KMeans(n_clusters = kmeans_n_clusters, random_state=seed)
    
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    plot_kmeans(kmeans_,Xtrain,kmeans_n_clusters,seed,ax)
    
# =============================================================================
#     ax.scatter(Xtrain[:,0],Xtrain[:,1],c=kmeans.labels_)
#     ax.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
# =============================================================================
    ax.grid()
    plt.title(f'Kmeans_Clusters_{dataSetName}')
    plt.savefig(f'Images/Kmeans_Clusters_{dataSetName}.png')
    plt.show()
    
    
    # Expectation Maximization
    vPrint('\tGMM: ')
    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance[1])
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        
        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))
        
    def plot_gmm(gmm, X, label=True, ax=None):
        ax = ax or plt.gca()
        labels = gmm.fit(X).predict(X)
        if label:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
        ax.axis('equal')
        gmm.covars_ = np.ones(gmm.means_.shape).T
        for i in range(gmm.covariances_.shape[-1]):
            for n in range(gmm.covariances_[:,:,i].shape[0]):
                gmm.covars_[i,n] = gmm.covariances_[:,:,i][n].mean()
                
        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)
            
    gmm_ss = []
    for n_f in n_clusters_:
        
        gmm = GaussianMixture(n_components = n_f,random_state=seed)
        gmm.fit(Xtrain)
        xgmm = gmm.fit_predict(Xtrain)
        ss_ = silhouette_score(Xtrain,gmm.predict(Xtrain))
        gmm_ss.append(ss_)
        vPrint(f'GMM Silhouette Scores: n_components = {n_f}; SS = {ss_}')
    
    
# =============================================================================
#         plot_gmm(gmm,Xtrain)
# =============================================================================
        
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    ax.plot(n_clusters_,gmm_ss)
    ax.set_xlabel('# of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.grid()
    plt.title(f'GMM_{dataSetName}_SilhouetteScore')
    plt.savefig(f'Images/GMM_{dataSetName}_SilhouetteScore.png')
    plt.show()
    
    
    # Plot all clusters
    u_l = np.unique(xgmm)
    #centroids = gmm.means_
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)

    for l in u_l:
        ax.scatter(Xtrain[xgmm == l][:,0],Xtrain[xgmm == l][:,1],c=Ytrain[xgmm == l])
    #ax.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    ax.grid()
    plt.title(f'GMM_Clusters_{dataSetName}')
    plt.savefig(f'Images/GMM_Clusters_{dataSetName}.png')
    plt.show()
    
    ## Determine optimal clusters
    try:
        dataset == ds1
        n_components = 200
        n_components_ = list(map(int,np.linspace(2,n_components,25)))
    except:
        n_components = 50
        n_components_ = list(map(int,np.linspace(2,n_components,25)))
        
    models = [GaussianMixture(n, covariance_type='full', random_state=seed).fit(Xtrain)
          for n in n_components_]
    
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    bics = [m.bic(Xtrain) for m in models]
    aics = [m.aic(Xtrain) for m in models]
    ax.plot(n_components_, bics, label='BIC')
    ax.plot(n_components_, aics, label='AIC')
    ax.legend(loc='best')
    ax.set_xlabel('n_components')
    plt.grid()
    fig.tight_layout()
    plt.title(f'GMM_AIC_BIC_{dataSetName}.png')
    plt.savefig(f'Images/GMM_AIC_BIC_{dataSetName}.png')
    plt.show()
    
    min_aic_components = n_components_[np.argmin(aics)]
    min_bic_components = n_components_[np.argmin(bics)]
    
    # plot min_aic_gmm
    gmm_aic = GaussianMixture(n_components = min_aic_components, random_state = seed)
    xgmm_aic = gmm_aic.fit_predict(Xtrain)
    
    gmm_bic = GaussianMixture(n_components = min_bic_components, random_state = seed)
    xgmm_bic = gmm_bic.fit_predict(Xtrain)
    
    fig, ax = plt.subplots(figsize = (8,6), dpi = 200)
    u_l_aic = np.unique(xgmm_aic)
    for l in u_l_aic:
        ax.scatter(Xtrain[xgmm_aic == l][:,0],Xtrain[xgmm_aic == l][:,1],c=Ytrain[xgmm_aic == l])
    
    ax.grid()
    plt.title(f'GMM_Clusters_min_aic{dataSetName}')
    plt.savefig(f'Images/GMM_Clusters_min_aic{dataSetName}.png')
    plt.show()
    
    fig, ax = plt.subplots(figsize = (8,6), dpi = 200)
    u_l_aic = np.unique(xgmm_bic)
    for l in u_l_aic:
        ax.scatter(Xtrain[xgmm_bic == l][:,0],Xtrain[xgmm_bic == l][:,1],c=Ytrain[xgmm_bic == l])
    
    ax.grid()
    plt.title(f'GMM_Clusters_min_bic{dataSetName}')
    plt.savefig(f'Images/GMM_Clusters_min_bic{dataSetName}.png')
    plt.show()
    
# =============================================================================
#         plot_results(Xtest, gmm.predict(Xtest), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")
# =============================================================================
###############################################################################
# re-apply clustering on dimension reduction data sets
###############################################################################
    # Pick best dr dataset clusters
    # Get best PCA 
    pca_target_cum_ev = 0.9
    xpca_cols = sum((cum_ev > pca_target_cum_ev) == False)
    pca_ = PCA(n_components = xpca_cols, random_state=seed)
    xpca_ = pca_.fit_transform(Xtrain)
    
    # Pick ica with max average kurtosis
    min_kurtosis = 3
    ica_n_comps = sum((np.array(ica_k_mean) > min_kurtosis) == False) + 1
    ica_ = FastICA(n_components = ica_n_comps, random_state=seed)
    xica_ = ica_.fit_transform(Xtrain)
    
    # Pick grp based on maximizing the ratio between Kurtosis / MSE
    try:
        dataset == ds1
        grp_n_comps = 13
    except:
        grp_n_comps = 10
        
# =============================================================================
#     grp_min_mse_i = np.argmin(np.array(grp_mse)[np.array(grp_n_features) <= n_features])
#     grp_n_comps = grp_n_features[grp_min_mse_i]
# =============================================================================
    grp_ = GaussianRandomProjection(n_components=grp_n_comps,random_state=seed,eps = 0.1)
    xgrp_ = grp_.fit_transform(Xtrain)
    
    # Set tsne
    tsne_ = tsne
    xtsne_ = xtsne
    
    dr_datasets = [xpca_, xica_, xgrp_, xtsne_]
    dr_names = [pca_, ica_, grp_, tsne_]
    
    # Clustering Algos
    # Pick best clustering algo setting
    # Pick Kmeans by silhouette score
    kmeans_n_clusters = n_clusters_[np.argmax(silhouette_scores)]
    kmeans_ = KMeans(n_clusters = kmeans_n_clusters, random_state=seed)
    
    # Pick GMM by minimum BIC
    gmm_ = gmm_bic
    
    cluster_algos = [kmeans_, gmm_]
    ### For each clustering algo, re-fit using DR datasets
    
    c_dr_ = {}
    c_a_n = ['KMeans','GMM']
    dr_n_ = ['PCA','ICA','GRP','t-SNE']
    fig, ax = plt.subplots(len(cluster_algos),len(dr_datasets),figsize=(10,8),dpi=200)
    for c, c_algo in enumerate(cluster_algos):
        c_dr_[c_algo] = {}
        for d, dr_data in enumerate(dr_datasets):
            dr_n = dr_names[d]
            # Fit
            xc_algo = c_algo.fit_predict(dr_data)
            
            c_dr_[c_algo][d] = xc_algo
            
            # Plot clusters
                    
            u_l = np.unique(xc_algo)
            for l in u_l:
                ax[c][d].scatter(dr_data[xc_algo == l][:,0],dr_data[xc_algo == l][:,1],c=Ytrain[xc_algo == l])
            
            ax[c][d].grid()
            ax[c][d].set_title(f'Clustering: {c_a_n[c]}\nDR: {dr_n_[d]}')
            
    fig.tight_layout()
    plt.savefig(f'Images/Clustering_and_DR_{dataSetName}.png')
    plt.show()
            

###############################################################################
# NEURAL NETWORK - APPLY DR
###############################################################################

## Setup vanilla NN for dataset
    max_iter = 1000
    ##Vary Learning_rate using learning_rate_init
    lr_rates = np.linspace(0.0005,0.001,10)
    n_h = list(map(int,np.linspace(10, 200,10)))
    estimator = MLPClassifier(max_iter = max_iter,
                              random_state = seed,
                              early_stopping = True,
                              )
    
    vPrint(f'Starting parameter grid search for {estimator}: {dataSetName}\n')
    
    param_grid = {
        'learning_rate_init':lr_rates,
        'activation':['relu'],
        'hidden_layer_sizes':n_h,
        }
    
    fold = []
    for i in range(Xdata.shape[0]):
        if Xdata[i] in Xtrain:
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
                                       #cv=ps,
                                       cv=2,
                                       )
    
    clf.fit(Xtrain,Ytrain)
    
    best_estimator_ = clf.best_estimator_
    
    vPrint(f'GridSearchCV Complete for {dataSetName} using {estimator}.')
    
    ####################
    # Retrain the best_estimator_ using pca, ica, rp, and manifold
    # PCA
    estimator_pca = MLPClassifier(max_iter = max_iter,
                                  random_state = seed,
                                  early_stopping = True,
                                  hidden_layer_sizes = best_estimator_.hidden_layer_sizes,
                                  learning_rate_init = best_estimator_.learning_rate_init,
                                  )
    estimator_ica = MLPClassifier(max_iter = max_iter,
                                  random_state = seed,
                                  early_stopping = True,
                                  hidden_layer_sizes = best_estimator_.hidden_layer_sizes,
                                  learning_rate_init = best_estimator_.learning_rate_init,
                                  )
    estimator_grp = MLPClassifier(max_iter = max_iter,
                                  random_state = seed,
                                  early_stopping = True,
                                  hidden_layer_sizes = best_estimator_.hidden_layer_sizes,
                                  learning_rate_init = best_estimator_.learning_rate_init,
                                  )
    estimator_tsne = MLPClassifier(max_iter = max_iter,
                                  random_state = seed,
                                  early_stopping = True,
                                  hidden_layer_sizes = best_estimator_.hidden_layer_sizes,
                                  learning_rate_init = best_estimator_.learning_rate_init,
                                  )
    
    estimator_pca.fit(xpca_,Ytrain)
    estimator_ica.fit(xica_,Ytrain)
    estimator_grp.fit(xgrp_,Ytrain)
    estimator_tsne.fit(xtsne_,Ytrain)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    ax.plot(best_estimator_.validation_scores_,label='Original')
    # PCA
    ax.plot(estimator_pca.validation_scores_,label='PCA')
    # ICA
    ax.plot(estimator_ica.validation_scores_,label='ICA')
    # GaussianRandomProjection RP
    ax.plot(estimator_grp.validation_scores_,label='GRP')
    # t-SNE
    ax.plot(estimator_tsne.validation_scores_,label='t-SNE')
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title(f'MLP_w_DR_{dataSetName}')
    fig.tight_layout()
    plt.savefig(f'Images/MLP_w_DR_{dataSetName}.png')
    plt.show()



###############################################################################
# NEURAL NETWORK - APPLY CLUSTERING
###############################################################################

    # For each clustering algorithm, apply clusters as new features for MLP
    # Use grid search
    xkmeans_ = kmeans_.fit_predict(Xtrain)
    Xtrain_c = np.append(Xtrain,xkmeans_.reshape(-1,1),axis=1) 

    xgmm_ = gmm_.fit_predict(Xtrain)
    Xtrain_c = np.append(Xtrain_c,xgmm_.reshape(-1,1),axis=1)
    
    # Setup new classifier
    mlp_c = MLPClassifier(max_iter = max_iter,
                                random_state = seed,
                                early_stopping = True,
                                )

    # Fit
    clf_c = model_selection.GridSearchCV(estimator=mlp_c, 
                                           param_grid = param_grid, 
                                           verbose=3 if verbose else 0,
                                           scoring = 'accuracy',
                                           n_jobs = n_jobs,
                                           return_train_score = True,
                                           #cv=ps,
                                           cv=2,
                                           )
    
    clf_c.fit(Xtrain_c,Ytrain)
    best_estimator_c = clf_c.best_estimator_
    
    
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    ax.plot(best_estimator_.validation_scores_,label='Original')
    ax.plot(best_estimator_c.validation_scores_,label='+ Clusters')
    ax.grid()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    plt.legend()
    plt.title(f'MLP_w_clustering_comparison_{dataSetName}')
    fig.tight_layout()
    plt.savefig(f'Images/MLP_w_clustering_comparison_{dataSetName}.png')
    plt.show()

    


































