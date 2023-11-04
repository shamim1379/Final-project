import pandas as pd
import pathlib as pth
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, silhouette_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


#reading CSV file by its path
my_csv = pth.Path("C:/Users/shamim/Desktop/Dataset Omar.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
#indicating input and target variables on file
X=df.iloc[:,1:4].values
y=df.iloc[:, 10].values.reshape(-1,1)
#scaling input data for KRR algorithm
sc=StandardScaler()
x_norm=sc.fit_transform(X)
#Split data to train and test part using normalized input data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x_norm, y, test_size = 0.25,random_state=10)
# Creating instance for KRR Algorithm
krr=KernelRidge()
#preparing parameters tuning for GBDT Algorithm
param_grid = {
    'alpha': [ 0.001,0.005,0.01,0.05,0.1,0.5, 1,5, 10,50],
    'kernel': ['additive_chi2', 'polynomial', 'linear', 'rbf', 'poly', 'chi2', 'sigmoid', 'laplacian', 'cosine'],
    'degree': [2, 3, 4,5],
    'gamma': [0.1,0.5, 1,5, 10],
    'coef0': [-1,0, 1, 2, 3],
}
#Tuning parameters with cross validation
grid_search = GridSearchCV(estimator=krr, param_grid=param_grid, cv=5 ,n_jobs=-1,scoring='r2')
#Training input data with target variable
grid_search.fit(X_Train, Y_Train)
#Selecting algorithm with best hyperparameters
best_krr=grid_search.best_estimator_
print(best_krr)
#Calculating R2 and MAE
Y_Pred_Train = best_krr.predict(X_Train)
Y_Pred_Test = best_krr.predict(X_Test)
R2_Train=r2_score(Y_Train,Y_Pred_Train)
R2_Test=r2_score(Y_Test,Y_Pred_Test)
MAE_Train=mean_absolute_error(Y_Train,Y_Pred_Train)
MAE_Test=mean_absolute_error(Y_Test,Y_Pred_Test)
#Printing Results
print("R2 for train set is: " , R2_Train)
print("R2 for test set is: " , R2_Test)
print("Mean absolute error for train part is: ",MAE_Train)
print("Mean absolute error for test part is: ",MAE_Test)