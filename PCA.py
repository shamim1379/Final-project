import pandas as pd
import pathlib as pth
import numpy as np
import sklearn as sk
from sklearn.metrics import r2_score, silhouette_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#Recieve file
my_csv = pth.Path("C:/Users/shamim/Desktop/Datasheet.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
#Seperating input data
X=df.iloc[:, :46].values
#Normalizing input data using standard scaler
sc = StandardScaler()
x_norm = sc.fit_transform(X)
#Create instance for PCA and fitting it
pca=PCA()
pca.fit(x_norm)
#Plotting explained variance of data vs Number of components
plt.plot(range(1,47), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

#Selecting optimum number for components and fit it using input data
pca_new=PCA(n_components=35)
pca_new.fit(x_norm)
pca_score=pca_new.transform(x_norm)
#Adding components to new excel file
df_kmean_insert = pd.concat([df.reset_index(drop=True), pd.DataFrame(pca_score)], axis=1)
df_kmean_insert.to_excel(excel_writer="C:/Users/shamim/Desktop/PCA score.xlsx")