# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# %%
df = pd.read_csv("model/Mall_Customers.csv")
df.head()

# %%
df.info()
df.describe()
df.isnull().sum()
df.columns


# %%
# Gender distribution
sns.countplot(data=df, x='Gender')
plt.title("Gender Distribution")

# Spending Score vs Income
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title("Spending Score vs Annual Income")
plt.show()


# %%
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# %%
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


# %%
k = 5  # Assume elbow shows 5
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)


# %%
df['Cluster'] = y_kmeans
df.head()


# %%
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=y_kmeans, palette='Set2', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.show()


# %%
df.groupby('Cluster')[['Annual Income (k$)','Spending Score (1-100)' ]].mean()


# %%
import joblib

# Save the trained KMeans model
joblib.dump(kmeans, 'model/kmeans_model.pkl')

# Save the scaler used for preprocessing
joblib.dump(scaler, 'model/scaler.pkl')

print("✅ KMeans model and scaler saved successfully!")
