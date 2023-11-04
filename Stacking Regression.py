import pandas as pd
import pathlib as pth
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, silhouette_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import  Ridge, Lasso
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import time

#Receive file
my_csv = pth.Path("C:/Users/shamim/Desktop/Dataset.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
# Select input and target variable for ML
X=df.iloc[:, :31].values
y=df.iloc[:, 31].values.reshape(-1,1)
#Normalizing Data
sc_train=StandardScaler()
x_norm=sc_train.fit_transform(X)
#Split data into train and test part
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x_norm, y, test_size = 0.25,random_state=10)
#Defining KRR, KNN , RF and GBDT algorithms with optimum parameters
rf = RandomForestRegressor(bootstrap= True, max_depth= 8,  max_features='auto', min_samples_leaf= 5, min_samples_split=2 , n_estimators=200)
krr = KernelRidge(alpha=0.05, coef0=-1, degree=2, gamma=0.1, kernel='polynomial')
knn=KNeighborsRegressor(algorithm='ball_tree', metric='manhattan',n_neighbors=23)
gbdt= GradientBoostingRegressor(learning_rate= 0.05, max_depth= 6, min_samples_split=3, n_estimators=150)
#Bulding our base models
base_models = [
    ('KNN', knn),
    ('RF',rf),
    ('KRR',krr),
    ('GBDT',gbdt),
    ]
#Using stack regressor to combine mentioned algorithms with cross validation
stacked = StackingRegressor(estimators = base_models,final_estimator = GradientBoostingRegressor(), cv = 5)
#Calculating computational time, MAE and R2 for all algorithms using a loop
for name, model in base_models:
    start_time = time.time()
    model.fit(X_Train, Y_Train)
    Y_Pred_Train=model.predict(X_Train)
    Y_Pred_Test = model.predict(X_Test)
    end_time = time.time()

    R2_Train = r2_score(Y_Train, Y_Pred_Train)
    R2_Test = r2_score(Y_Test, Y_Pred_Test)
    MAE_Train = mean_absolute_error(Y_Train, Y_Pred_Train)
    MAE_Test = mean_absolute_error(Y_Test, Y_Pred_Test)
    print("-------{}-------".format(name))
    print("R2 for train set is: ", R2_Train)
    print("R2 for test set is: ", R2_Test)
    print("Mean absolute error for train part is: ", MAE_Train)
    print("Mean absolute error for test part is: ", MAE_Test)
    print("Computation Time: {}".format(end_time - start_time))
    print("----------------------------------\n")
#Calculating final results for stacing algorithm
start_times = time.time()
stacked.fit(X_Train, Y_Train)
Y_Pred_Trains=stacked.predict(X_Train)
Y_Pred_Tests = stacked.predict(X_Test)
end_times = time.time()
R2_Trains = r2_score(Y_Train, Y_Pred_Trains)
R2_Tests = r2_score(Y_Test, Y_Pred_Tests)
MAE_Trains = mean_absolute_error(Y_Train, Y_Pred_Trains)
MAE_Tests = mean_absolute_error(Y_Test, Y_Pred_Tests)
print("-------Stacked Ensemble-------")
print("R2 for train set is: ", R2_Trains)
print("R2 for test set is: ", R2_Tests)
print("Mean absolute error for train part is: ", MAE_Trains)
print("Mean absolute error for test part is: ", MAE_Tests)
print("Computation Time: {}".format(end_times - start_times))
print("----------------------------------")