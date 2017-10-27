from sklearn import linear_model
from sklearn import tree
from scipy.linalg import norm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas as pd 
import numpy as np 
from numpy import median
import math
import time
import os, psutil

 
def whichAlgorithm(x):
    return {
        0:'Linear Regression' ,
        1:'Bayesian Ridge',
        2:'SGD Regression',
        3:'Decision Tree Regression'
    }[x]

def norm(arr):
    oldMax = max(arr)
    oldMin = min(arr)
    oldRange = oldMax - oldMin
    newMin = 0
    newMax = 1
    normalisedArray = np.array([(newMin + (((x-oldMin)*(newMax-newMin)))/(oldMax - oldMin)) for x in arr])
    return normalisedArray   


def runProgram(given, pred, kFoldVariable, algo):
        start = time.time()
       
        explainedVarianceScore = cross_val_score(algo,  given, pred, cv=kFoldVariable, scoring= 'explained_variance')
        explainedVarianceFinal = (norm(explainedVarianceScore)).mean()
        print("Explained Variance Score: ", (explainedVarianceFinal)) 
        
        meanAbsoluteError = cross_val_score(algo,  given, pred, cv=kFoldVariable, scoring= 'neg_mean_absolute_error')
        meanAbsoluteErrorNorm = norm(meanAbsoluteError)
        meanAbsoluteErrorFinal = abs(meanAbsoluteErrorNorm.mean())
        print("Mean Absolute Error: ", (meanAbsoluteErrorFinal))
       
        meanSquaredError = cross_val_score(algo,  given, pred, cv=kFoldVariable, scoring= 'neg_mean_squared_error')
        meanSquaredErrorNorm = norm(meanSquaredError)
        meanSquaredErrorFinal = abs(meanSquaredErrorNorm.mean())
        print("Mean Squared Error: ", meanSquaredErrorFinal)
  
        medianAbsoluteError = cross_val_score(algo,  given, pred, cv=kFoldVariable, scoring= 'neg_median_absolute_error')
        medianAbsoluteErrorNorm = norm(medianAbsoluteError)
        medianAbsoluteErrorFinal = abs(median(medianAbsoluteErrorNorm))
        print("Median Absolute Error: ", medianAbsoluteErrorFinal)

        r2 = cross_val_score(algo,  given, pred, cv=kFoldVariable, scoring='r2')
        r2Norm = norm(r2)
        r2Final = r2Norm.mean()
        print("r2: ", r2Final)

        elapsed = time.time()-start
        print("Time elapsed:", elapsed)
        
        


kF= KFold(n_splits=10)
newsFile = "news.csv"
news = pd.read_csv(newsFile, header=0)
news = news.drop('url',  axis = 1)
g = np.array(news)
g = g[:, :-1]
p = np.array(news)
p = p[:, -1]


lnr = linear_model.LinearRegression()
rdg = linear_model.BayesianRidge()
sgd = linear_model.SGDRegressor()
dtr = tree.DecisionTreeRegressor()
i=0 
print("Current Algorithm Running: ", whichAlgorithm(i))
runProgram(g, p, kF, lnr)
i+=1
print("Current Algorithm Running: ", whichAlgorithm(i))
runProgram(g, p, kF, rdg)
i+=1
print("Current Algorithm Running: ", whichAlgorithm(i))
runProgram(g, p, kF, sgd)
i+=1
print("Current Algorithm Running: ", whichAlgorithm(i))
runProgram(g, p, kF, dtr)


#process = psutil.Process(os.getpid())
#process.get_ext_memory_info().peak_wset
#print p.get_memory_percent()



