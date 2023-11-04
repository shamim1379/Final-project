import pandas as pd
import pathlib as pth
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, silhouette_score,mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

my_csv = pth.Path("C:/Users/shamim/Desktop/PCA-S.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
for n in range(15,30):
    X = df.iloc[:, :n].values
    y = df.iloc[:, 30].values.reshape(-1, 1)
    print(X)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.25, random_state=1)
    gbdt = GradientBoostingRegressor(learning_rate= 0.05, max_depth= 6, min_samples_split= 3, n_estimators=150)
    gbdt.fit(X_Train,Y_Train)
    Y_Pred_Train = gbdt.predict(X_Train)
    Y_Pred_Test = gbdt.predict(X_Test)
    R2_Train = r2_score(Y_Train, Y_Pred_Train)
    R2_Test = r2_score(Y_Test, Y_Pred_Test)
    MAE_Train = mean_absolute_error(Y_Train, Y_Pred_Train)
    MAE_Test = mean_absolute_error(Y_Test, Y_Pred_Test)
    print("for number of components of: ", n )
    print("R2 for train set is: ", R2_Train)
    print("R2 for test set is: ", R2_Test)
    print("Mean absolute error for train part is: ", MAE_Train)
    print("Mean absolute error for test part is: ", MAE_Test)
    print("-----------------------------------------")