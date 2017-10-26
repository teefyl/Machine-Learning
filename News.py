from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from scipy.linalg import norm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from numpy import median
import math


def runProgram(rows, cols, kFoldVariable):
    lnr = linear_model.LinearRegression()
    rdg = linear_model.BayesianRidge()
    sgd = linear_model.SGDRegressor()
    clf = tree.DecisionTreeRegressor()
    

    algorithm= np.array([lnr, rdg, sgd,  clf])
    for i in range(0, len(algorithm)):

        print("Current Algorithm: ", str(algorithm[i]))

       
        explainedVarianceScore = cross_val_score(algorithm[i],  rows, cols, cv=kFoldVariable, scoring= 'explained_variance')
        print("Explained Variance Score: ", explainedVarianceScore) 

        
        meanAbsoluteError = cross_val_score(algorithm[i],  rows, cols, cv=kFoldVariable, scoring= 'neg_mean_absolute_error')
        meanAbsoluteErrorFinal = abs(meanAbsoluteError.mean())
        print("Mean Absolute Error: ", meanAbsoluteErrorFinal )

       
        meanSquaredError = cross_val_score(algorithm[i],  rows, cols, cv=kFoldVariable, scoring= 'neg_mean_squared_error')
        meanSquaredErrorFinal = abs(meanSquaredError.mean())
        print("Mean Squared Error: ", meanSquaredErrorFinal)

        
        medianAbsoluteError = cross_val_score(algorithm[i],  rows, cols, cv=kFoldVariable, scoring= 'neg_median_absolute_error')
        medianAbsoluteErrorFinal = abs(median(medianAbsoluteError))
        print("Median Absolute Error: ", medianAbsoluteErrorFinal)

        
        r2 = cross_val_score(algorithm[i],  rows, cols, cv=kFoldVariable, scoring='r2')
        print("r2: ", r2)







newsFile = "news.csv"
news = pd.read_csv(newsFile, header=0)
news = news.drop('url',  axis = 1)

#print(news.columns)
houseSalesFiles= "house.csv"
houseSales = pd.read_csv(houseSalesFiles, header=0)
houseSales.drop('id', axis=1)


R1 = np.array(news)
R1 = R1[:, :-1]
C1 = np.array(news)
C1 = C1[:, -1]
R1 = R1.astype('int') ##all data in columns are ints
C1 = C1.astype('int')


kF= KFold(n_splits=10)

print("News")
runProgram(R1, C1, kF)










#raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
#       'last_name': ['Miller', 'Jacobson', ".", 'Milner', 'Cooze'], 
#      'age': [42, 52, 36, 24, 73], 
#     'preTestScore': [4, 24, 31, ".", "."],
#    'postTestScore': ["25,000", "94,000", 57, 62, 70]}



#df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])
#df.to_csv('"task3, team_46, cherry evaluation, data.csv"')
#predicted = cross_val_predict(lr, news.data, y, cv=10)
#fig, ax = plt.subplots()
#ax.scatter(y, predicted, edgecolors=(0, 0, 0))
#ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#plt.show()

