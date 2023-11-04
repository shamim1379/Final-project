import pandas as pd
import pathlib as pth
import numpy as np
import sklearn as sk
from sklearn.metrics import r2_score, silhouette_score,mean_absolute_error
from sklearn.cluster import KMeans



my_csv = pth.Path("C:/Users/shamim/Desktop/FP.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
X=df.iloc[:,:31].values
print(X)
for i in range(2, 26):
    kmeans = KMeans(n_clusters=i,n_init=10, init="k-means++")
    kmeans.fit(X)
    label = kmeans.predict(X)
    print('Silhouette Score(n= {} ): '.format(i), silhouette_score(X, label))