import pandas as pd
import pathlib as pth
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

#reading CSV file by its path
my_csv = pth.Path("C:/Users/shamim/Desktop/Dataset Omar.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
plt.figure(figsize=(50,30))
cor=df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()