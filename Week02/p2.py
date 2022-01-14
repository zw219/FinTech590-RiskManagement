import numpy as np
from numba import njit
import pandas as pd
import statsmodels.formula.api as sm
from math import *
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import seaborn as sns

# read file and store
file = pd.read_csv("problem2.csv")

x = file.x
y = file.y

results = sm.ols(formula="y ~ x", data=file).fit()
results.summary()
beta = results.params[1]
intercept = results.params[0]
err = []
for i in range(0, len(file)):
    e = y[i] - beta * x[i] - intercept
    err.append(e)

sns.kdeplot(err)