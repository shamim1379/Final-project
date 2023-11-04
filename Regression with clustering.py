import pandas as pd
import pathlib as pth
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, silhouette_score,mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import  Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Receive file
my_csv = pth.Path("C:/Users/shamim/Desktop/Dataset.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
# Select input and target variable for ML
X = df.iloc[:, :31].values
y = df.iloc[:, 31].values.reshape(-1, 1)

# Initialize the range of clusters
n_clusters = range(2, 31)

# Initialize the list for mean absolute errors
mae_Train = []
mae_Test = []

# Get the total size of the dataset
total_size = X.shape[0]

# Loop through each number of clusters
for n in n_clusters:
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Initialize the total mean absolute error
    total_mae_Train = 0
    total_mae_Test = 0

    # Loop through each cluster
    for cluster in range(n):
        # Get the data points belonging to the current cluster
        X_cluster = X[clusters == cluster]
        y_cluster = y[clusters == cluster]

        # Get the size of the current cluster
        cluster_size = X_cluster.shape[0]

        # Split the data into train and test parts
        X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.25, random_state=42)

        # Create an instance for GBDT with optimum hyperparameters
        grid_search = GradientBoostingRegressor(random_state=1, learning_rate= 0.05, max_depth= 6, min_samples_split= 3, n_estimators=150)
        grid_search.fit(X_train, y_train)

        # calculate MAE for test and train parts
        y_pred_train = grid_search.predict(X_train)
        y_pred_test = grid_search.predict(X_test)
        train_mae_pred = mean_absolute_error(y_train, y_pred_train)
        test_mae_pred = mean_absolute_error(y_test, y_pred_test)

        # Add the current cluster's MAE (weighted by cluster size) to the total
        total_mae_Train += train_mae_pred * cluster_size
        total_mae_Test += test_mae_pred * cluster_size

    # Calculate the total MAE
    avg_mae_train = total_mae_Train / total_size
    print(avg_mae_train)
    mae_Train.append(avg_mae_train)
    avg_mae_test = total_mae_Test / total_size
    print(avg_mae_test)
    mae_Test.append(avg_mae_test)
# Print MAE value for each cluster
print(mae_Train)
print(mae_Test)
#plot results
plt.plot(range(2,31),np.c_[mae_Train,mae_Test])
plt.xlabel('Number of cluster')
plt.ylabel('MAE (V)')
plt.title('Train and test MAE (V)')
plt.show()