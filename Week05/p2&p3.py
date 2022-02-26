from math import *
from numpy import inf
from numpy import copy
from numpy.linalg import norm
from numpy import linalg as LA
from datetime import datetime
from random import *
import math
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.optimize import minimize
import numpy as np


class MyClass:
    def __init__(self, port_path, price_path):
        port = pd.read_csv(port_path)
        prices = pd.read_csv(price_path)
        portA = port.loc[port['Portfolio'] == "A"][["Stock", "Holding"]]
        portA.Holding = portA.Holding / sum(portA.Holding)
        portB = port.loc[port['Portfolio'] == "B"][["Stock", "Holding"]]
        portB.Holding = portB.Holding / sum(portB.Holding)
        portC = port.loc[port['Portfolio'] == "C"][["Stock", "Holding"]]
        portC.Holding = portC.Holding / sum(portC.Holding)
        port = port[["Stock", "Holding"]]
        port.Holding = port.Holding / sum(port.Holding)
        portA = portA.set_index('Stock')
        portB = portB.set_index('Stock')
        portC = portC.set_index('Stock')
        port = port.set_index('Stock')
        self.port = port
        self.portA = portA
        self.portB = portB
        self.portC = portC
        pricesA = prices[list(portA.index)]
        pricesB = prices[list(portB.index)]
        pricesC = prices[list(portC.index)]
        prices = pd.concat([pricesA, pricesB, pricesC])
        returnsA = pricesA.pct_change()[1:]
        returnsB = pricesB.pct_change()[1:]
        returnsC = pricesC.pct_change()[1:]
        returns = prices.pct_change()[1:]
        returnsA.to_csv("returnA.csv", index=False)
        returnsB.to_csv("returnB.csv", index=False)
        returnsC.to_csv("returnC.csv", index=False)
        returns.to_csv("returns.csv", index=False)
        covA = returnsA.cov()
        covB = returnsB.cov()
        covC = returnsC.cov()
        cov = returns.cov()
        meanA = returnsA.mean()
        meanB = returnsB.mean()
        meanC = returnsC.mean()
        meanT = returns.mean()
        portA_mean = meanA.dot(np.array(portA.Holding))
        portB_mean = meanB.dot(np.array(portB.Holding))
        portC_mean = meanC.dot(np.array(portC.Holding))
        port_mean = meanT.dot(np.array(port.Holding))
        # Calculate portfolio standard deviation
        portA_stdev = np.sqrt(np.array(portA.Holding).T.dot(covA).dot(np.array(portA.Holding)))
        portB_stdev = np.sqrt(np.array(portB.Holding).T.dot(covB).dot(np.array(portB.Holding)))
        portC_stdev = np.sqrt(np.array(portC.Holding).T.dot(covC).dot(np.array(portC.Holding)))
        port_stdev = np.sqrt(np.array(port.Holding).T.dot(cov).dot(np.array(port.Holding)))
        # Select our confidence interval (I'll choose 95% here)
        conf_level1 = 0.05
        cutoffA = norm.ppf(conf_level1, portA_mean, portA_stdev)
        cutoffB = norm.ppf(conf_level1, portB_mean, portB_stdev)
        cutoffC = norm.ppf(conf_level1, portC_mean, portC_stdev)
        cutoffT = norm.ppf(conf_level1, port_mean, port_stdev)
        self.varA = 1 - cutoffA
        self.varB = 1 - cutoffB
        self.varC = 1 - cutoffC
        self.varT = 1 - cutoffT

    # Q1 a
    def get_exp_weighted_cov_matrix(self, path):
        seed(1)
        returns = pd.read_csv(path)
        returns.drop(returns.columns[0], axis=1, inplace=True)
        means = returns.mean()
        size = len(returns.columns)
        returns.cov().to_csv('regular_cov.csv', index=False)
        returns.corr().to_csv('regular_corr.csv', index=False)

        def cov(weights, i, j):
            mean_i = means[i]
            mean_j = means[j]
            col_i = returns.iloc[:, i]
            col_j = returns.iloc[:, j]
            cov_list = weights * (col_i - mean_i) * (col_j - mean_j)
            return sum(cov_list)

        def eigengraph(λ):
            weights = [(1 - λ) * λ ** (i - 1) for i in range(1, len(returns.index) + 1)]
            weights = [weight / sum(weights) for weight in weights]
            cov_matrix = [[cov(weights, i, j) for i in range(size)] for j in range(size)]
            e_value, e_vector = LA.eig(np.array(cov_matrix))
            e_value.sort()
            e_value = e_value[::-1]
            cumulative_var = np.array([sum(e_value[:i]) / sum(e_value) for i in range(1, len(e_value) + 1)],
                                      dtype="complex_")
            plt.plot([i for i in range(len(e_value))], cumulative_var)
            plt.title('λ = ' + str(λ) + ' cumulative eigenvalue graph')
            plt.ylabel('cumulative variance')
            plt.xlabel('K value')
            plt.show()
            return cov_matrix

        c1 = pd.DataFrame(eigengraph(0.97), columns=returns.columns, index=returns.columns)
        corr_matrix = [[0 for i in range(len(c1))] for j in range(len(c1))]
        for i in range(len(c1)):
            for j in range(i, len(c1)):
                if i == j:
                    corr_matrix[i][j] = 1.0
                    continue
                cov = c1.iloc[i][j]
                stdi = math.sqrt(c1.iloc[i][i])
                stdj = math.sqrt(c1.iloc[j][j])
                corr_matrix[i][j] = cov / stdi / stdj
                corr_matrix[j][i] = corr_matrix[i][j]
        corr_matrix = pd.DataFrame(corr_matrix, columns=returns.columns, index=returns.columns)
        corr_matrix.to_csv("97_corr.csv", index=False)
        c1.to_csv("97_cov.csv", index=False)
        return c1

    def get_reg_cov_matrix(path):
        returns = pd.read_csv(path)
        return returns.cov()

    # Q2
    # https://stackify.dev/670594-how-can-i-calculate-the-nearest-positive-semi-definite-matrix
    def nearPSD(A, n, epsilon=0.0):
        eigval, eigvec = np.linalg.eig(A)
        val = np.matrix(np.maximum(eigval, epsilon))
        vec = np.matrix(eigvec)
        T = 1 / (np.multiply(vec, vec) * val.T)
        T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
        B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
        out = B * B.T
        return np.asarray(out)

    # https://github.com/mikecroucher/nearest_correlation/blob/master/nearest_correlation.py
    def nearcorr(self, A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
                 weights=None, verbose=False,
                 except_on_too_many_iterations=True):

        if (isinstance(A, ValueError)):
            ds = copy(A.ds)
            A = copy(A.matrix)
        else:
            ds = np.zeros(np.shape(A))

        eps = np.spacing(1)
        if not np.all((np.transpose(A) == A)):
            raise ValueError('Input Matrix is not symmetric')
        if not tol:
            tol = eps * np.shape(A)[0] * np.array([1, 1])
        if weights is None:
            weights = np.ones(np.shape(A)[0])
        X = copy(A)
        Y = copy(A)
        rel_diffY = inf
        rel_diffX = inf
        rel_diffXY = inf

        Whalf = np.sqrt(np.outer(weights, weights))

        iteration = 0
        while max(rel_diffX, rel_diffY, rel_diffXY) > tol[0]:
            iteration += 1
            if iteration > max_iterations:
                if except_on_too_many_iterations:
                    if max_iterations == 1:
                        message = "No solution found in " \
                                  + str(max_iterations) + " iteration"
                    else:
                        message = "No solution found in " \
                                  + str(max_iterations) + " iterations"
                    raise ValueError(message, X, iteration, ds)
                else:

                    return X

            Xold = copy(X)
            R = X - ds
            R_wtd = Whalf * R
            if flag == 0:
                X = self.proj_spd(R_wtd)
            elif flag == 1:
                raise ValueError("Setting 'flag' to 1 is currently\
                                     not implemented.")
            X = X / Whalf
            ds = X - R
            Yold = copy(Y)
            Y = copy(X)
            np.fill_diagonal(Y, 1)
            normY = norm(Y, 'fro')
            rel_diffX = norm(X - Xold, 'fro') / norm(X, 'fro')
            rel_diffY = norm(Y - Yold, 'fro') / normY
            rel_diffXY = norm(Y - X, 'fro') / normY
            X = copy(Y)
        return X

    def proj_spd(self, A):
        d, v = np.linalg.eigh(A)
        A = (v * np.maximum(d, 0)).dot(v.T)
        A = (A + A.T) / 2
        return (A)

    # Q3
    def simulation(self, path):
        self.get_exp_weighted_cov_matrix(path)

        def chol_psd(root, sigma, n):
            root = [[0 for i in range(n)] for j in range(n)]
            sigma = np.array(sigma)
            for j in range(n):
                s = 0.0
                if j > 0:
                    s = sum([root[j][i] ** 2 for i in range(j)])
                temp = sigma[j][j] - s
                if temp <= 0:
                    temp = 0.0
                root[j][j] = np.sqrt(temp)
                if 0.0 == root[j][j]:
                    for i in range(j + 1, n):
                        root[j][i] = 0.0
                else:
                    ir = 1.0 / root[j][j]
                    for i in range(j + 1, n):
                        # may be wrong
                        s = sum([root[i][k] * root[j][i] for k in range(j)])
                        root[i][j] = (sigma[i][j] - s) * ir
            return root

        def simulateNormal(self, cap_n, cov, mean=[], seed=1234):
            n, m = len(cov), len(cov[0])
            if n != m:
                raise Exception()
                return
            np.random.seed = seed
            temp_mean = [0.0 for i in range(n)]
            # maybe wrong
            m = len(mean)
            if m != 0:
                if n != m:
                    raise Exception()
                else:
                    temp_mean = mean

            root = np.empty([n, cap_n])
            root = chol_psd(root, cov, n)

            mu, sigma = 0.0, 1.0  # mean and standard deviation
            out = np.random.normal(mu, sigma, (n, cap_n))

            out = np.matmul(root, out).transpose()

            for j in range(n):
                for i in range(cap_n):
                    out[i][j] = out[i][j] + temp_mean[j]
            return out

        def simulate_pca(self, a, nsim, pctExp=1, mean=[], seed=1234):
            n = len(a)
            temp_mean = [0.0 for i in range(n)]
            if len(mean) > 0:
                temp_mean = mean
            vals, vecs = np.linalg.eig(np.array(a))
            #     少一步 vals = real.(vals)
            posv = [idx for idx in range(len(vals)) if vals[idx] > 0]
            vals.sort()
            vals = vals[::-1]
            tv = sum(vals)
            if pctExp < 1:
                nval = 0
                pct = 0.0
                for i in range(len(posv)):
                    pct += vals[i] / tv
                    nval += 1
                    if pct >= pctExp:
                        break
                if nval < len(posv):
                    posv[:] = posv[0: nval]
            tempvecs = []
            for vec in vecs:
                tempvec = []
                for idx in posv:
                    tempvec.append(vec[idx])
                tempvecs.append(tempvec)
            vals = vals[:len(posv)]
            vecs = tempvecs
            #     B = vecs * np.diag(np.array(np.sqrt(vals)))
            B = np.matmul(vecs, np.diag(np.array(np.sqrt(vals))))
            np.random.seed = seed
            m = len(vals)
            #     randn(m, nsim)没懂
            r = np.random.rand(m, nsim)
            out = np.matmul(B, r).transpose()
            for j in range(m):
                for i in range(n):
                    out[i][j] = out[i][j] + temp_mean[i]
            return out

        covar = pd.read_csv("97_cov.csv")
        covar = np.array(covar)
        # sim = simulateNormal(101, covar)
        # sim = simulate_pca(covar,101,pctExp=.5)
        pearson_cov = pd.read_csv("regular_cov.csv")
        pearson_std = pearson_cov.transform(lambda x: x ** 0.5)
        pearson_cor = pd.read_csv("regular_corr.csv")
        ewma_cov = pd.read_csv("97_cov.csv")
        ewma_std = ewma_cov.transform(lambda x: x ** 0.5)
        ewma_cor = pd.read_csv("97_corr.csv")
        matrixType = ["EWMA", "EWMA_COR_PEARSON_STD", "PEARSON", "PEARSON_COR_EWMA_STD"]
        simType = ["Full", "PCA=1", "PCA=0.75", "PCA=0.5"]
        matrixLookup = {}
        matrixLookup["EWMA"] = ewma_cov
        matrixLookup["EWMA_COR_PEARSON_STD"] = np.diag(np.array(pearson_std)) * ewma_cor * np.diag(
            np.array(pearson_std))
        matrixLookup["PEARSON"] = pearson_cov
        matrixLookup["PEARSON_COR_EWMA_STD"] = np.diag(np.array(ewma_std)) * pearson_cor * np.diag(np.array(ewma_std))

        matrix = [""] * 16
        simulation = [""] * 16
        runtimes = [""] * 16
        norms = [""] * 16

        i = 0
        for sim in simType:
            for mat in matrixType:
                matrix[i] = mat
                simulation[i] = sim
                c = np.array(matrixLookup[mat])
                elapse = 0.0
                s = []
                st = datetime.now()
                if sim == "Full":
                    for loops in range(20):
                        s = simulateNormal(25000, c).transpose()
                elif sim == "PCA=1":
                    for loops in range(20):
                        s = simulate_pca(c, 25000, pctExp=1).transpose()
                elif sim == "PCA=0.75":
                    for loops in range(20):
                        s = simulate_pca(c, 25000, pctExp=.75).transpose()
                else:
                    for loops in range(20):
                        s = simulate_pca(c, 25000, pctExp=.5).transpose()
                elapse = (datetime.now() - st) / 20
                covar = np.cov(s)

                runtimes[i] = elapse
                norms[i] = sum(np.dot(covar - c, covar - c))
                i = i + 1
        outTable = pd.DataFrame(data={'matrix': matrix, 'simulation': simulation, 'runtimes': runtimes, 'norms': norms})
        return outTable

    # Q4
    def norm_var(self, path, ticker):
        returns = pd.read_csv(path)
        ticker = returns[ticker]
        tick_mean = ticker.mean()
        tick_demean = ticker - tick_mean
        # normal dist from here
        std = np.std(tick_demean)
        var_n = 1.65 * std
        return var_n

    def mleTVaR(self, path, ticker, alpha):
        returns = pd.read_csv(path)
        ret = returns[ticker]
        mu_norm, std_norm = norm.fit(ret)

        def negLogLikeForT(initialParams):
            df, sigma = initialParams
            return -t(df=df, scale=sigma).logpdf(ret).sum()

        initialParams = np.array([1, 1])
        df, sigma = minimize(negLogLikeForT, initialParams, method="BFGS").x
        return -t.ppf(alpha, df, loc=ret.mean(), scale=sigma)

    # Historical var
    def VaR_hist(self, path, Confidence_Interval=0.95, Period_Interval=None,
                 Series=False, removeNa=True):
        Returns = pd.read_csv(path)
        Returns.drop(Returns.columns[0], axis=1, inplace=True)
        if removeNa == True: Returns = Returns[pd.notnull(Returns)]

        if (Series == True and Period_Interval == None):
            Period_Interval = 100
        elif Period_Interval == None:
            Period_Interval = len(Returns)

        if Series == False:
            Data = Returns[-Period_Interval:]
            Value_at_Risk = -np.percentile(Data, 1 - Confidence_Interval)
        if Series == True:
            Value_at_Risk = pd.Series(index=Returns.index, name='HSVaR')
            for i in range(0, len(Returns) - Period_Interval):
                if i == 0:
                    Data = Returns[-(Period_Interval):]
                else:
                    Data = Returns[-(Period_Interval + i):-i]
                Value_at_Risk[-i - 1] = -np.percentile(Data, 1 - Confidence_Interval)
        return (Value_at_Risk)

    # Q5
    def ES_n(self, path, ticker, alpha):
        returns = pd.read_csv(path)
        ret = returns[ticker]
        mu_norm, std_norm = norm.fit(ret)
        return alpha ** -1 * norm.pdf(norm.ppf(alpha)) * std_norm - mu_norm

    def ES_t(self, path, ticker, alpha):
        returns = pd.read_csv(path)
        p1 = returns[ticker]
        mu_norm, std_norm = norm.fit(p1)
        xanu = t.ppf(alpha, (len(p1.index) - 1))
        return -1 / alpha * (1 - (len(p1.index) - 1)) ** (-1) * ((len(p1.index) - 1) - 2 + xanu ** 2) * t.pdf(xanu, (
                    len(p1.index) - 1)) * std_norm - mu_norm

    def VaR_n_port(self):
        return "VaR_A: " + str(self.varA) + "\n" + "VaR_B: " + str(self.varB) + "\n" + "VaR_C: " + str(self.varC) + "\n" + "VaR_Total: " + str(self.varT) + "\n"+"\n"

    def Es_t_port(self, alpha):
        returns = pd.read_csv("returns.csv")
        mean = returns.mean()
        std = returns.std()
        xanu = t.ppf(alpha, (len(returns.index) - 1))
        p_return = -1 / alpha * (1 - (len(returns.index) - 1)) ** (-1) * ((len(returns.index) - 1) - 2 + xanu ** 2) * t.pdf(xanu, (
                len(returns.index) - 1)) * std - mean
        arr1 = np.array(self.port["Holding"])
        arr2 = np.array(p_return)
        return sum([arr1[i] * arr2[i] for i in range(len(arr1))])

    def Es_t_portA(self, alpha):
        returns = pd.read_csv("returnA.csv")
        mean = returns.mean()
        std = returns.std()
        xanu = t.ppf(alpha, (len(returns.index) - 1))
        p_return = -1 / alpha * (1 - (len(returns.index) - 1)) ** (-1) * ((len(returns.index) - 1) - 2 + xanu ** 2) * t.pdf(xanu, (
                len(returns.index) - 1)) * std - mean
        arr1 = np.array(self.portA["Holding"])
        arr2 = np.array(p_return)
        return sum([arr1[i] * arr2[i] for i in range(len(arr1))])

    def Es_t_portB(self, alpha):
        returns = pd.read_csv("returnB.csv")
        mean = returns.mean()
        std = returns.std()
        xanu = t.ppf(alpha, (len(returns.index) - 1))
        p_return = -1 / alpha * (1 - (len(returns.index) - 1)) ** (-1) * ((len(returns.index) - 1) - 2 + xanu ** 2) * t.pdf(xanu, (
                len(returns.index) - 1)) * std - mean
        arr1 = np.array(self.portB["Holding"])
        arr2 = np.array(p_return)
        return sum([arr1[i] * arr2[i] for i in range(len(arr1))])

    def Es_t_portC(self, alpha):
        returns = pd.read_csv("returnC.csv")
        mean = returns.mean()
        std = returns.std()
        xanu = t.ppf(alpha, (len(returns.index) - 1))
        p_return = -1 / alpha * (1 - (len(returns.index) - 1)) ** (-1) * ((len(returns.index) - 1) - 2 + xanu ** 2) * t.pdf(xanu, (
                len(returns.index) - 1)) * std - mean
        arr1 = np.array(self.portC["Holding"])
        arr2 = np.array(p_return)
        return sum([arr1[i] * arr2[i] for i in range(len(arr1))])

a = MyClass("portfolio.csv","DailyPrices.csv")
print("ES_A: " + str(a.Es_t_portA(0.05)))
print("ES_B: " + str(a.Es_t_portB(0.05)))
print("ES_C: " + str(a.Es_t_portC(0.05)))
print("ES_total: " + str(a.Es_t_port(0.05)) + "\n")
print(a.VaR_n_port())
