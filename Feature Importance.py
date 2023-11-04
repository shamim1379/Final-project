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
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
#Receive file
my_csv = pth.Path("C:/Users/shamim/Desktop/Dataset Omar.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
# indicating input and target variables
X=df.iloc[:,:10]
y=df.iloc[:, 10].values.reshape(-1,1)
print(X)
print(y)
#split data to test and trin
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.25, random_state = 1)
#creating instance for GBDT/RF algorithm
gbdt=GradientBoostingRegressor()
#Algorithm training
gbdt.fit(X_Train, Y_Train)
#printing importance for each feature in input data
for feature,importance in zip(X.columns,gbdt.feature_importances_):
    print(f'{feature}:{importance}')