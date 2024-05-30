from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import itertools
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp

##### Configuration
n_clusters = 1000  # You can adjust this number based on your needs
do_PCA = False
dim_pca = 30
viz = False
cluster_target = ['product', 'customer']


#### step 1 : clustering product by color, size, group
# Read the dataset from a CSV file
file_path = './etail/root/task1_data.txt'  # Update this path to your actual file location
df_org = pd.read_csv(file_path)
df = df_org.copy()
features = ['order', 'product', 'customer', 'color', 'size', 'group']
feature_dim = [df[feature].nunique() for feature in features]

for target in cluster_target:
    target_onehot = np.zeros((df[target].nunique(), feature_dim[3] + feature_dim[4] + feature_dim[5]))

    for feature in features[3:]:
        for i, fes in enumerate(df.groupby(target)[feature].unique()):
            target_onehot[i, fes] = 1



    #### step 2 : clustering customer by color, size, group
    # product_onehot : dim(product) x dim(color) + dim(size) + dim(group)

    # Apply clustering algorithm (e.g., KMeans)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(target_onehot)

    # Create a DataFrame for products with their cluster labels
    target_cluster_df = pd.DataFrame({target: df[target].unique(), 'cluster': clusters})

    if viz:
        # Optional: Analyze the clustering results
        # Plot the distribution of products in each cluster
        plt.figure(figsize=(10, 6))
        sns.countplot(x='cluster', data=target_cluster_df)
        plt.title('Distribution of target in Each Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of target')
        plt.show()

    df[target] = df[target].map(target_cluster_df.set_index(target)['cluster'])


    if viz:
        # visualize the clustering results by plotting the products in a 2D space
        # Apply PCA to reduce the dimensionality of the product features
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        target_pca = pca.fit_transform(target_onehot)

        # Create a DataFrame for the PCA-transformed product features
        target_pca_df = pd.DataFrame(target_pca, columns=['PC1', 'PC2'])
        target_pca_df['cluster'] = clusters

        # Plot the products in a 2D space with different colors for each cluster
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=target_pca_df, palette='viridis')
        plt.title('Clustering of Products in a 2D Space')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.show()

    print(f'Clustering of {target} is completed.')

if do_PCA == False:
    output = np.zeros((df['order'].nunique(), n_clusters*2 + dim_pca*2))

    n_column = 0
    for target in cluster_target:
        for i, fes in enumerate(df.groupby('order')[target].unique()):
            output[i, n_column+fes] = 1
        n_column += df[target].nunique()
        print(n_column)


    # np.save('task1_one_hot.npy', output)

    output = sp.csr_matrix(output)
    import pickle
    with open("etail/ours/clustering_onehot.pickle", 'wb') as f:
        pickle.dump(output, f)

    breakpoint()

else:
    raise NotImplementedError('PCA is not implemented yet.')
    output = np.zeros((df['order'].nunique(), n_clusters*2))