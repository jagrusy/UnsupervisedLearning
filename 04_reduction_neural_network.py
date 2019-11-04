import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import FastICA, PCA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection

from sklearn.datasets import load_digits
from util import getCreditCardData, getWineData

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def dim_reduction_nn(X, y, reducer, reducer_name):
    # Define a pipeline to search for the best combination of reducer truncation
    # and classifier regularization.
    nn = MLPClassifier(alpha=0.235, momentum=0.9, random_state=3)

    pipe = Pipeline(steps=[('reducer', reducer), ('nn', nn)])

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'reducer__n_components': [1,2,3,6,12,18,22],
        'nn__hidden_layer_sizes': [(10,)]
    }
    search = GridSearchCV(pipe, param_grid, iid=False, cv=5, n_jobs=-1)
    search.fit(X, y)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    # Plot the reducer spectrum
    reducer.fit(X)
    
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.set_title('Dimensionality Reduced Data for NN (using {})'.format(reducer_name))

    if hasattr(reducer, 'explained_variance_ratio_'):
        ax0.plot(reducer.explained_variance_ratio_, linewidth=2)
        ax0.set_ylabel('reducer explained variance')
    elif hasattr(reducer, 'noise_variance_'):
        ax0.plot(reducer.noise_variance_, linewidth=2)
        ax0.set_ylabel('reducer noise variance')
    elif hasattr(reducer, 'mean_'):
        ax0.plot(reducer.mean_, linewidth=2)
        ax0.set_ylabel('reducer means')
        ax0.set_yscale('log')
    else:
        m1 = np.mean(reducer.components_, axis=1)
        ax0.plot(np.mean(reducer.components_, axis=0), linewidth=2)
        ax0.set_ylabel('reducer component means')

    ax0.axvline(search.best_estimator_.named_steps['reducer'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(path_or_buf='Figs/04_{}_dim_reduction_nn.csv'.format(reducer_name))
    components_col = 'param_reducer__n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                legend=False, ax=ax1)
    ax1.set_ylabel('Classification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.tight_layout()
    plt.savefig('Figs/04_{}_dim_reduction_nn'.format(reducer_name))

np.random.seed(0)
X, y, data = getCreditCardData('./Data/ccdefault.xls')

reducer = PCA(random_state=0)
dim_reduction_nn(X, y, reducer, 'pca')

reducer = FastICA(random_state=0)
dim_reduction_nn(X, y, reducer, 'ica')

reducer = GaussianRandomProjection(n_components=22, random_state=0)
dim_reduction_nn(X, y, reducer, 'random_projection')

reducer = FactorAnalysis(random_state=0)
dim_reduction_nn(X, y, reducer, 'factor_analysis')
