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
X=df.iloc[:, :7].values
y=df.iloc[:,7].values.reshape(-1,1)
#scaling input data for KRR algorithm
sc_train=StandardScaler()
x_norm=sc_train.fit_transform(X)
mean=sc_train.mean_
std=sc_train.scale_
#Split data to train and test part using normalized input data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x_norm, y, test_size = 0.25,random_state=10)
#Defining an instance using optimum hyperparameters
krr = KernelRidge(alpha=0.1,kernel='poly',gamma=0.1,coef0=3,degree=2)
#fiting input data with output data using algorithm
krr.fit(X_Train, Y_Train)
#Calculating R2 and MAE with comparing algorithm and dataset answers
Y_Pred_Train = krr.predict(X_Train)
Y_Pred_Test = krr.predict(X_Test)
R2_Train=r2_score(Y_Train,Y_Pred_Train)
R2_Test=r2_score(Y_Test,Y_Pred_Test)
MAE_Train=mean_absolute_error(Y_Train,Y_Pred_Train)
MAE_Test=mean_absolute_error(Y_Test,Y_Pred_Test)
#Printing results
print("R2 for train set is: " , R2_Train)
print("R2 for test set is: " , R2_Test)
print("Mean absolute error for train part is: ",MAE_Train)
print("Mean absolute error for test part is: ",MAE_Test)
#reading test part dataset
test_csv = pth.Path("C:/Users/shamim/Desktop/Test.csv")
dt = pd.read_csv(test_csv.resolve(), sep=',')
#indicating input dataset for predictions part
Z=dt.iloc[:,:7].values
z_norm=(Z-mean)/std
#Obtaining answers
prediction=krr.predict(z_norm)
#printing predictions
print(prediction)