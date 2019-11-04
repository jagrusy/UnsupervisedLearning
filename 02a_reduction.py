import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from yellowbrick.features.pca import PCADecomposition

from sklearn.datasets import load_digits
from util import getCreditCardData, getWineData

def kurtosis(x):
    n = np.shape(x)[0]
    mean = np.sum((x**1)/n) # Calculate the mean
    var = np.sum((x-mean)**2)/n # Calculate the variance
    skew = np.sum((x-mean)**3)/n # Calculate the skewness
    kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = kurt/(var**2)-3
    return [kurt, skew, var, mean]

def dim_reduction(X, y, target_names, data_set, n_components, dim):
    pca = PCA(n_components=n_components)
    X_r = pca.fit(X).transform(X)
    
    kt = np.array([0,0,0,0,0])
    for i in range(1,10):
        ica = FastICA(n_components=i)
        X_r1 = ica.fit(X).transform(X)
        vals = kurtosis(X_r1)
        vals.append(i)
        kt = np.vstack((kt, vals))
    df = pd.DataFrame(kt, columns=['Kurtosis', 'Skew', 'Variance', 'Mean', 'N Components'])
    df.to_csv('Figs/02a_ica_kurt_{}.csv'.format(data_set))

    fca = FactorAnalysis(n_components=n_components)
    X_r2 = fca.fit(X).transform(X)

    rtx = GaussianRandomProjection(n_components=n_components)
    X_r3 = rtx.fit(X).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
        % str(pca.explained_variance_ratio_))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    lw = 2 # line width
    outputs = np.unique(y)

    plt.figure()
    for color, i, target_name in zip(colors, outputs, target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, dim], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of {} dataset'.format(data_set))
    plt.savefig('Figs/02_PCA of {} dataset'.format(data_set))

    plt.figure()
    for color, i, target_name in zip(colors, outputs, target_names):
        plt.scatter(X_r1[y == i, 0], X_r1[y == i, dim], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('ICA of {} dataset'.format(data_set))
    plt.savefig('Figs/02_ICA of {} dataset'.format(data_set))

    plt.figure()
    for color, i, target_name in zip(colors, outputs, target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, dim], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Factor Analysis of {} dataset'.format(data_set))
    plt.savefig('Figs/02_FactorAnalysis of {} dataset'.format(data_set))

    plt.figure()
    for color, i, target_name in zip(colors, outputs, target_names):
        plt.scatter(X_r3[y == i, 0], X_r3[y == i, dim], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Random Projection of {} dataset'.format(data_set))
    plt.savefig('Figs/02_Random Projection of {} dataset'.format(data_set))

np.random.seed(0)

digits = load_digits()
X, y = digits.data, digits.target
dim_reduction(X, y, target_names=range(0,10), data_set='Digits', n_components=2, dim=1)

X, y, data = getCreditCardData('./Data/ccdefault.xls')
dim_reduction(X, y, target_names=['paid', 'defaulted'], data_set='Credit', n_components=2, dim=1)

# X, y = digits.data, digits.target
# dim_reduction(X, y, target_names=range(0,10), data_set='Digits', n_components=2, dim=1)

# plt.clf()
# visualizer = PCADecomposition(scale=True, projection=2, proj_features=True, heatmap=True)
# visualizer.fit_transform(X, y)
# visualizer.show()
# plt.clf()

# X, y, data = getCreditCardData('./Data/ccdefault.xls')
# dim_reduction(X, y, target_names=['paid', 'defaulted'], data_set='Credit', n_components=2, dim=1)

# visualizer = PCADecomposition(scale=True, projection=2, proj_features=True, heatmap=True)
# visualizer.fit_transform(X, y)
# visualizer.show()
# plt.clf()