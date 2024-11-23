# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

Step 1. Start the program

Step 2. Import the necessary python libraries

Step 3. Read the dataset of Mall_Customers csv file

Step 4. From sklearn libraary select the cluster and import KMeans Clustering

Step 5. Find the sum of squared distance between each points and the centroid in a cluster using Elbow Method

Step 6. Plot the graph x and y as Number of Clusters and wcss respectively

Step 7. Using the matplotlib library draw the scatter plot for the given number of clusters (ie. here n_clusters = 5)

Step 8. Stop the program

## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Harish Ragav S
RegisterNumber: 212222110013
```
```
import pandas as pd
df=pd.read_csv("Mall_Customers.csv")
df.head()
df.info()
df.isnull().sum()
from sklearn.cluster import KMeans
wcss=[] #within-CLuster sum of square
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(df.iloc[:,3:])
    wcss.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters = 5)
km.fit(df.iloc[:,3:])
y_pred = km.predict(df.iloc[:,3:])
y_pred
df["cluster"] = y_pred
a = df[df["cluster"]==0]
b = df[df["cluster"]==1]
c = df[df["cluster"]==2]
d = df[df["cluster"]==3]
e = df[df["cluster"]==4]
plt.scatter(a["Annual Income (k$)"],a["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(b["Annual Income (k$)"],b["Spending Score (1-100)"],c="blue",label="cluster1")
plt.scatter(c["Annual Income (k$)"],c["Spending Score (1-100)"],c="black",label="cluster2")
plt.scatter(d["Annual Income (k$)"],d["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(e["Annual Income (k$)"],e["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:

![Screenshot 2024-10-05 114548](https://github.com/user-attachments/assets/eb70b89f-03c1-4459-a8b9-4ac55afa0c9e)

![Screenshot 2024-10-05 114617](https://github.com/user-attachments/assets/ed1c5603-ff78-4866-adcb-04537c54e1df)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
