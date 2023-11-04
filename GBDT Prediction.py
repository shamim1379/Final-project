import pandas as pd
import pathlib as pth
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

#Reading train dataset file from files by its path
train_csv = pth.Path("C:/Users/shamim/Desktop/Train.csv")
df = pd.read_csv(train_csv.resolve(), sep=',')
#Indicating input and target variables on file
X=df.iloc[:, 0:7].values
y=df.iloc[:, 7].values.reshape(-1,1)
#split data to train and test parts
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.25, random_state = 31)
#defining GBDT algorithm with optimum hyperparameters which has been obtained from tuning part
gbdt = GradientBoostingRegressor(learning_rate= 0.2, max_depth= 2, min_samples_split= 3, n_estimators= 150)
#fiting input data with output data using algorithm
gbdt.fit(X_Train, Y_Train)
#predicting algorithm's answers for train and test part
Y_Pred_Train = gbdt.predict(X_Train)
Y_Pred_Test = gbdt.predict(X_Test)
#Calculating R2 and MAE with comparing algorithm and dataset answers
R2_Train=r2_score(Y_Train,Y_Pred_Train)
R2_Test=r2_score(Y_Test,Y_Pred_Test)
MAE_Train=mean_absolute_error(Y_Train,Y_Pred_Train)
MAE_Test=mean_absolute_error(Y_Test,Y_Pred_Test)
#printing R2 and MAE result for train and test
print("R2 for train set is: " , R2_Train)
print("R2 for test set is: " , R2_Test)
print("Mean absolute error for train part is: ",MAE_Train)
print("Mean absolute error for test part is: ",MAE_Test)
#reading test part dataset
test_csv = pth.Path("C:/Users/shamim/Desktop/Test.csv")
dt = pd.read_csv(test_csv.resolve(), sep=',')
#indicating input dataset for predictions part
Z=dt.iloc[:,:7].values
#predict answers for new test data
prediction=gbdt.predict(Z)
#printing predictions
print(prediction)