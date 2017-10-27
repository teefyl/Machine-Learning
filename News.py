from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd 
import numpy as np 
from numpy import median
import math, time, os, psutil

listTimes = [] #global list of times
i=0            #global i variable which will be used for the below switch statement


#Which Algorithm is Running Function
#Parameters : int X
#Returns : String (that states which function is running)
def whichAlgorithm(x): 
    return {
        0:'Bayesian Ridge',
        1:'Linear Regression' ,
        2:'SGD Regression',
        3:'Decision Tree Regression'
    }[x]

#Normalisation Function
#Parameters: Array that needs to be normalised
#Returns : Array that's been normalised
#Method : Uses the formula 
def norm(arr): 
    oldMax = max(arr)
    oldMin = min(arr)
    oldRange = oldMax - oldMin
    newMin = 0
    newMax = 1
    normalisedArray = np.array([(newMin + (((x-oldMin)*(newMax-newMin)))/(oldMax - oldMin)) for x in arr])
    return normalisedArray   

#Algorithm Program
#Parameters : Given data array, pred array, the K Fold Variable(in this case 10), and the algorithm to run it with.
#Returns : Void
#Calculates each individual metric using the algorithm
def runProgram(given, pred, kFoldVariable, algo):
        global i
        print("Current Algorithm Running: ", whichAlgorithm(i)) #Prints which algorithm is currently running
        start = time.time() #stores the start time of this sequence
       
        explainedVarianceScore = cross_val_score(algo,  given, pred, cv=kFoldVariable, scoring= 'explained_variance') #calculates array containing the explainedVarianceScores
        explainedVarianceFinal = (norm(explainedVarianceScore)).mean() #normalises the array and then gets the mean
        print("Explained Variance Score: ", (explainedVarianceFinal))  #prints the mean 
        
        meanAbsoluteError = cross_val_score(algo,  given, pred, cv=kFoldVariable, scoring= 'neg_mean_absolute_error') #calculates negative array
        meanAbsoluteErrorNorm = norm(meanAbsoluteError) #normalises the array
        meanAbsoluteErrorFinal = abs(meanAbsoluteErrorNorm.mean()) #gets the absolute value of the mean
        print("Mean Absolute Error: ", (meanAbsoluteErrorFinal)) #prints the absolute value of the mean 
       
        meanSquaredError = cross_val_score(algo,  given, pred, cv=kFoldVariable, scoring= 'neg_mean_squared_error') #calculates negative array 
        meanSquaredErrorNorm = norm(meanSquaredError) #normalises the array
        meanSquaredErrorFinal = abs(meanSquaredErrorNorm.mean()) #gets the abs value of the mean
        print("Mean Squared Error: ", meanSquaredErrorFinal)     #prints the absolute value of the mean
  
        medianAbsoluteError = cross_val_score(algo,  given, pred, cv=kFoldVariable, scoring= 'neg_median_absolute_error') #calculates neg array
        medianAbsoluteErrorNorm = norm(medianAbsoluteError) #normalises the array
        medianAbsoluteErrorFinal = abs(median(medianAbsoluteErrorNorm)) #gets the abs value of the median
        print("Median Absolute Error: ", medianAbsoluteErrorFinal) #prints this value

        r2 = cross_val_score(algo,  given, pred, cv=kFoldVariable, scoring='r2') #calculates the array of R2 values
        r2Norm = norm(r2) #normalises the array
        r2Final = r2Norm.mean() #gets the mean of this array 
        print("r2: ", r2Final) #prints the mean of the normalised r2 array

        elapsed = time.time()-start #current runtime = current time - start time
        global listTimes
        listTimes.append(elapsed) #adding the runtime to the list of times
        i+=1
         
rdg = linear_model.BayesianRidge()
lnr = linear_model.LinearRegression()
sgd = linear_model.SGDRegressor()
dtr = tree.DecisionTreeRegressor()

kF= KFold(n_splits=10) # sets the k value for our k fold validation to 10 
newsFile = "news.csv"  # the first datasets local path
news = pd.read_csv(newsFile, header=0, nrows = 20000) #Cutting off the dataset at 20000 rows in order to keep the sizes the same
#Dropping the following columns for containing letters and not numbers
news = news.drop('url',  axis = 1)
g = np.array(news) # given data array
g = g[:, :-1]
p = np.array(news) # predicted data array
p = p[:, -1]

i=0
print("ONLINE NEWS DATASET")
runProgram(g, p, kF, rdg) #Running the Online News Dataset With Bayesian Ridge
runProgram(g, p, kF, lnr) #Running the Online News Dataset with Linear Regression
runProgram(g, p, kF, sgd) #Running the Online News Dataset with SGD Regression
runProgram(g, p, kF, dtr) #Running the Online News Dataset with Decision Tree Regression

houseSalesFiles= "house.csv" #the second datasets local path
houseSales = pd.read_csv(houseSalesFiles, header=0, nrows = 20000) #Cutting off the dataset at 20000 rows in order to keep the sizes the same

#Dropping the following columns for containing letters and not numbers
houseSales = houseSales.drop('id', axis=1)
houseSales = houseSales.drop('price', axis=1)
houseSales = houseSales.drop('date', axis=1)

g = np.array(houseSales)
g = g[:, :-1]
p = np.array(houseSales)
p = p[:, -1]


i=0 
print("HOUSE SALES DATASET")
runProgram(g, p, kF, rdg)   #Running the House Sales Dataset With Bayesian Ridge
runProgram(g, p, kF, lnr)   #Running the House Sales Dataset with Linear Regression 
runProgram(g, p, kF, sgd)   #Running the House Sales Dataset with SGD Regression 
runProgram(g, p, kF, dtr)   #Running the House Sales Dataset with Decision Tree Regression

print ("LISTS OF TIMES" ,listTimes[0:8]) #All runtimes in seconds, for each algorithm
arrayTimes = np.asarray(listTimes) #Converting the list of Runtimes to an Array of Runtimes for Normalisation
normTimes = norm(arrayTimes) #Normalising the Array of Runtimes
print("LISTS OF NORMALISED TIMES ", normTimes[0:8]) #Array Of Normalised Runtimes 

##Redundant code that I was planning on using for getting memory usage. 
#process = psutil.Process(os.getpid())
#process.get_ext_memory_info().peak_wset
#print p.get_memory_percent()