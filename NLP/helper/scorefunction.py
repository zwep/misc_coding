# encoding: utf-8

"""
Here we show how we defined the score function for several models..

"""

import sklearn
from helper.plotfunction import tsne_plot
import numpy as np

def score_logistic_regression(features, target, feature_label, target_label=None, plot=False, p_train=0.7):
    """
    Simple function for plotting and scoring a simple linear regression.

    :param features:
    :param target:
    :param feature_label:
    :param target_label:
    :param plot:
    :param p_train:
    :return:
    """

    if target_label is None:
        target_label = target

    n = len(features)
    n_train = int(np.round(n*p_train))
    index_full = range(n)
    index_train = np.random.choice(index_full, n_train, replace=False)
    index_test = list(set.difference(set(index_full), set(index_train)))

    if plot:
        tsne_plot(features, target_label)

    model = sklearn.linear_model.logistic.LogisticRegression()
    linear_reg = model.fit(features[index_train], target[index_train])
    test_score = linear_reg.score(features[index_test], target[index_test])
    train_score = linear_reg.score(features[index_train], target[index_train])

    beta_hat = linear_reg.coef_[0, :]
    beta_hat = [abs(x) for x in beta_hat]

    n_max = 10
    print(np.array(feature_label)[beta_hat.argsort()[-n_max:][::-1]])

    return test_score, train_score

