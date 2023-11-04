import pandas as pd
import pathlib as pth
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, silhouette_score,mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import GridSearchCV


#reading CSV file by its path
my_csv = pth.Path("C:/Users/shamim/Desktop/Dataset Omar.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
#indicating input and target variables on file
X=df.iloc[:,1:4].values
y=df.iloc[:, 10].values.reshape(-1,1)
#Split data to train and test part
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.25, random_state = 1)
# Creating instance for GBDT Algorithm
gbdt=GradientBoostingRegressor()
#preparing parameters tuning for GBDT Algorithm
param_grid = {
    'n_estimators': [50,150,250,350],
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 3, 4,5],
    'learning_rate': [0.01,0.05,0.1,0.15,0.2]
}
#Tuning parameters with cross validation
grid_search = GridSearchCV(estimator=gbdt, param_grid=param_grid,cv=5, n_jobs=-1)
grid_search.fit(X_Train, Y_Train)
#printing best hyperparameters
print(grid_search.best_params_)
best_gbdt=grid_search.best_estimator_
# calculating MAE and R-square for dataset
Y_Pred_Train = best_gbdt.predict(X_Train)
Y_Pred_Test = best_gbdt.predict(X_Test)
R2_Train=r2_score(Y_Train,Y_Pred_Train)
R2_Test=r2_score(Y_Test,Y_Pred_Test)
MAE_Train=mean_absolute_error(Y_Train,Y_Pred_Train)
MAE_Test=mean_absolute_error(Y_Test,Y_Pred_Test)
#printing results
print("R2 for train set is: " , R2_Train)
print("R2 for test set is: " , R2_Test)
print("Mean absolute error for train part is: ",MAE_Train)
print("Mean absolute error for test part is: ",MAE_Test)