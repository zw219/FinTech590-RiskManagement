import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from math import *

import statistics
import seaborn as sns
from scipy.optimize import minimize
from numpy.linalg import norm

# read file and store
file = pd.read_csv("problem2.csv")

x = file.x
y = file.y

#OLS distribution
results = sm.ols(formula="y ~ x", data=file).fit()
results.summary()
beta = results.params[1]
intercept = results.params[0]
err = []
for i in range(0, len(file)):
    e = y[i] - beta * x[i] - intercept
    err.append(e)

sns.kdeplot(err)

varx = statistics.variance(x)
vary = statistics.variance(y)
# cited from https://analyticsindiamag.com/maximum-likelihood-estimation-python-guide/
def MLE_Norm(parameters):
    # extract parameters
    const, beta, std_dev = parameters
    # predict the output
    pred = const + beta*x
    # Calculate the log-likelihood for normal distribution
    LL = np.sum(norm.logpdf(y, pred, std_dev))
    # Calculate the negative log-likelihood
    neg_LL = -1*LL
    return neg_LL

mle_model = minimize(MLE_Norm, np.array([intercept,beta,sqrt(vary)]), method='L-BFGS-B')
mle_model

