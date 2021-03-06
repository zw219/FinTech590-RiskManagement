import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from math import *


# read file and store
# switch columns between x and y
file = pd.read_csv("problem1.csv")
file.insert(0, 'y', file.pop('y'))

x = np.array(file.x)
y = np.array(file.y)
miu = np.array([x.mean(), y.mean()])
sig = np.array(file.cov())

beta = sig[0][1]/sig[1][1]
intercept = miu[0] * (-1) * miu[1]
std = sqrt(sig[0][0] - pow(sig[0][1],2) / sig[1][1])

# OLS

results = sm.ols(formula="y ~ x", data=file).fit()
results.summary()

# Compare beta
print("Conditional Distribution beta: " + str(beta))
print("Ordinary Least Squares beta: " + str(results.params[1]))

# compare intercept
print("Conditional Distribution beta: " + str(intercept))
print("Ordinary Least Squares beta: " + str(results.params[0]))