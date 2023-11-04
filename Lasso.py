import pandas as pd
import pathlib as pth
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.linear_model import LinearRegression

#reading CSV file by its path
my_csv = pth.Path("C:/Users/shamim/Desktop/Dataset Omar.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
#indicating input and target variables on file
X=df.iloc[:,3].values.reshape(-1,1)
y=df.iloc[:, 10].values.reshape(-1,1)
#Split data to train and test part
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.25, random_state = 1)
#Creating list of alpha for Lasso regression tuning
param_grid = {'alpha':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,3,4,5,10,50,100]}
#Create instance of Lasso algorithm with tuning and fit them using data of training part
lasso = Lasso()
lasso_search = GridSearchCV(lasso, param_grid, cv=5)
lasso_search.fit(X_Train, Y_Train)
#Printing best alpha
print("Best alpha for lasso regression:", lasso_search.best_params_['alpha'])
#Obtaining results in terms of MAE and R2 and printing them
Y_Pred_Train = lasso_search.predict(X_Train)
Y_Pred_Test = lasso_search.predict(X_Test)
R2_Train=r2_score(Y_Train,Y_Pred_Train)
R2_Test=r2_score(Y_Test,Y_Pred_Test)
MAE_Train=mean_absolute_error(Y_Train,Y_Pred_Train)
MAE_Test=mean_absolute_error(Y_Test,Y_Pred_Test)
print("R2 for train set is: " , R2_Train)
print("R2 for test set is: " , R2_Test)
print("Mean absolute error for train part is: ",MAE_Train)
print("Mean absolute error for test part is: ",MAE_Test)