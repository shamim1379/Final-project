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


#Comments are same with KRR
my_csv = pth.Path("C:/Users/shamim/Desktop/Dataset.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
X=df.iloc[:, :31].values
y=df.iloc[:, 31].values.reshape(-1,1)
sc=StandardScaler()
x_norm=sc.fit_transform(X)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x_norm, y, test_size = 0.25,random_state=10)
knn=KNeighborsRegressor()
param_grid = {'n_neighbors': range(2,31),
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan','minkowski'],
              'leaf_size': [5,10,15,20,25,30],
              'algorithm': ['auto','brute','ball_tree','kd_tree']}

grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3 ,n_jobs=-1)
grid_search.fit(X_Train, Y_Train)
best_knn=grid_search.best_estimator_
print(best_knn)
Y_Pred_Train = best_knn.predict(X_Train)
Y_Pred_Test = best_knn.predict(X_Test)
R2_Train=r2_score(Y_Train,Y_Pred_Train)
R2_Test=r2_score(Y_Test,Y_Pred_Test)
MAE_Train=mean_absolute_error(Y_Train,Y_Pred_Train)
MAE_Test=mean_absolute_error(Y_Test,Y_Pred_Test)
print("R2 for train set is: " , R2_Train)
print("R2 for test set is: " , R2_Test)
print("Mean absolute error for train part is: ",MAE_Train)
print("Mean absolute error for test part is: ",MAE_Test)