import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, scale
import matplotlib.pyplot as plt
import pickle

def save_cv(cv_results, alg_name, data_name):
    pickle_out = open("Pickles/{}-{}.pickle".format(alg_name, data_name),"wb")
    pickle.dump(cv_results, pickle_out)
    pickle_out.close()

def getCreditCardData(path, subset=1.0):
    print('reading credit card data')
    df = pd.read_excel(io=path, header=1)
    df = df.apply(pd.to_numeric, errors='coerce')
    df['MARRIAGE']  = np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])
    df['EDUCATION'] = np.where(df['EDUCATION'] == 5, 4, df['EDUCATION'])
    df['EDUCATION'] = np.where(df['EDUCATION'] == 6, 4, df['EDUCATION'])
    df['EDUCATION'] = np.where(df['EDUCATION'] == 0, 4, df['EDUCATION'])
    df['PAY_0'] = np.where(df['PAY_0'] <= 0, -1, df['PAY_0'])
    df['PAY_2'] = np.where(df['PAY_2'] <= 0, -1, df['PAY_2'])
    df['PAY_3'] = np.where(df['PAY_3'] <= 0, -1, df['PAY_3'])
    df['PAY_4'] = np.where(df['PAY_4'] <= 0, -1, df['PAY_4'])
    df['PAY_5'] = np.where(df['PAY_5'] <= 0, -1, df['PAY_5'])
    df['PAY_6'] = np.where(df['PAY_6'] <= 0, -1, df['PAY_6'])  
    df = df.drop(columns=['ID'])
    df = df.sample(frac=subset, random_state=0)
    # df = df.drop(columns=['PAY_0',  'PAY_2',  'PAY_3',  'PAY_4',  'PAY_5',  'PAY_6','BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',  'BILL_AMT4',  'BILL_AMT5',  'BILL_AMT6',  'PAY_AMT1',  'PAY_AMT2',  'PAY_AMT3',  'PAY_AMT4',  'PAY_AMT5',  'PAY_AMT6'])
    # df = df.drop(columns=['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'])
    # print(df.describe())
    # print(df.head())
    # Include all columns except the last
    X = df.iloc[:,:-2].values

    # Select last column
    y = df.iloc[:,-1].values
    a,b = np.unique(y, return_counts=True)
    print('data contains {} on time payments and {} defaults'.format(b[0], b[1]))

    return X, y, scale(df.values)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=3)

    # return X_train, X_test, y_train, y_test

def getWineData(path, test_size=0.2):
    print('reading wine data')
    df = pd.read_csv(filepath_or_buffer=path, sep=';')
    df = df.apply(pd.to_numeric, errors='coerce')

    # Include all columns except the last
    X = df.iloc[:,:-2].values

    # Select last column
    y = df.iloc[:,-1].values
    # print('unique y values are: {}'.format(np.unique(y)))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=3)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X, y, df
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.fit_transform(X_test)
    # return X_train, X_test, y_train, y_test


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_fitness_curve(x_axis, curve_rhc, curve_sa, curve_ga, curve_mim, title):
    plt.figure()
    plt.title('{} vs Iterations'.format(title))
    plt.xlabel("Iterations")
    plt.ylabel(title)
    plt.xscale('log')

    plt.grid()

    plt.plot(x_axis, curve_mim, color="k",
             label="MIMIC")
    plt.plot(x_axis, curve_rhc, color="r",
             label="Random Hill Climbing")
    plt.plot(x_axis, curve_sa, color="g",
             label="Simulated Annealing")
    plt.plot(x_axis, curve_ga, color="b",
             label="Genetic Algorithm")

    plt.legend(loc="best")
    return plt

def plot_param_curve(x_axis, curve, title, param):
    plt.figure()
    plt.title('{} vs {}'.format(title, param))
    plt.xlabel(param)
    plt.ylabel(title)
    # plt.xscale('log')

    plt.grid()

    plt.plot(x_axis, curve, color="k",
             label='{} with {}'.format(title, param))

    plt.legend(loc="best")
    return plt

def plot_temp_curve(x_axis, curve_exp, curve_art, title):
    plt.figure()
    plt.title('Fitness vs Initial Temperature for Continuous Peaks')
    plt.xlabel("Initial Temperature")
    plt.ylabel(title)
    # plt.xscale('log')

    plt.grid()

    plt.plot(x_axis, curve_exp, color="g",
             label='Exponential Decay')

    plt.plot(x_axis, curve_art, color="r",
             label='Arithmetic Decay')

    plt.legend(loc="best")
    return plt

