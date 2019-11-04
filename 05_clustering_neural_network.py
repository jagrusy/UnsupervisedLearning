import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_digits
from util import getCreditCardData, getWineData

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

class custGMM(GaussianMixture):
    def transform(self, X):
        return self.predict_proba(X)

def cluster_nn(X, y, reducer, reducer_name):
    # Define a pipeline to search for the best combination of reducer truncation
    # and classifier regularization.
    nn = MLPClassifier(max_iter=1000, alpha=0.235, momentum=0.9, random_state=3)

    pipe = Pipeline(steps=[('reducer', reducer), ('nn', nn)])

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    if (reducer_name == 'kmeans'):
        param_grid = {
            'reducer__n_clusters': [1,2,3,6,12,18,22],
            'nn__hidden_layer_sizes': [(10,)]
        }
        components_col = 'param_reducer__n_clusters'
    else:
        param_grid = {
            'reducer__n_components': [1,2,3,6,9,10],
            'nn__hidden_layer_sizes': [(10,)]
        }
        components_col = 'param_reducer__n_components'
    search = GridSearchCV(pipe, param_grid, iid=False, cv=5, n_jobs=-1)
    search.fit(X, y)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(6, 6))
    ax.set_title('Dimensionality Reduced Data for NN (using {})'.format(reducer_name))

    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(path_or_buf='Figs/05_{}_cluster_nn.csv'.format(reducer_name))
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                legend=False, ax=ax)
    ax.set_ylabel('Classification accuracy (val)')
    ax.set_xlabel('n_components')

    plt.tight_layout()
    plt.savefig('Figs/05_{}_cluster_nn'.format(reducer_name))

np.random.seed(0)
X, y, data = getCreditCardData('./Data/ccdefault.xls')

reducer = KMeans(random_state=0)
cluster_nn(X, y, reducer, 'kmeans')

reducer = custGMM(GaussianMixture(reg_covar=1.0, random_state=0))
cluster_nn(X, y, reducer, 'gaussian_mix')

