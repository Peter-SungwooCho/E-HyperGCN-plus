from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import itertools
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from sklearn.decomposition import PCA

##### Configuration
n_clusters = 2000  # You can adjust this number based on your needs
do_PCA = True
dim_pca = 30
viz = False
cluster_target = ['product', 'customer']

n_cluster_2 = 1000
two_step_clustering = True
memo = f'_task2_n_cluster({n_clusters})_n_cluster_2({n_cluster_2})_dim_pca({dim_pca})'


#### step 1 : clustering product by color, size, group
# Read the dataset from a CSV file
file_path = './task2_data.txt'  # Update this path to your actual file location
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

if two_step_clustering:
    df_2 = df_org.copy()

    for target in cluster_target:
        tmp_df = df_org.copy()

        target_onehot = np.zeros((tmp_df[target].nunique(), n_clusters + feature_dim[3] + feature_dim[4] + feature_dim[5]))
        
        other_target = [t for t in cluster_target if t != target][0]
        tmp_df[other_target] = df[other_target]

        tmp_features = [other_target, 'color', 'size', 'group']
        for feature in tmp_features:
            for i, fes in enumerate(tmp_df.groupby(target)[feature].unique()):
                target_onehot[i, fes] = 1



        #### step 2 : clustering customer by color, size, group
        # product_onehot : dim(product) x dim(color) + dim(size) + dim(group)

        # Apply clustering algorithm (e.g., KMeans)
        kmeans = KMeans(n_clusters=n_cluster_2, random_state=42)
        clusters = kmeans.fit_predict(target_onehot)

        # Create a DataFrame for products with their cluster labels
        target_cluster_df = pd.DataFrame({target: tmp_df[target].unique(), 'cluster': clusters})

        if viz:
            # Optional: Analyze the clustering results
            # Plot the distribution of products in each cluster
            plt.figure(figsize=(10, 6))
            sns.countplot(x='cluster', data=target_cluster_df)
            plt.title('Distribution of target in Each Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Number of target')
            plt.show()

        df_2[target] = df_2[target].map(target_cluster_df.set_index(target)['cluster'])


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
    output = np.zeros((df['order'].nunique(), n_clusters*2))

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


else:
    final_df = df if two_step_clustering == False else df_2
    output = np.zeros((df['order'].nunique(), n_clusters*2))

    n_column = 0
    for target in cluster_target:
        for i, fes in enumerate(final_df.groupby('order')[target].unique()):
            output[i, n_column+fes] = 1
        n_column += final_df[target].nunique()
        print(n_column)


    final_n_cluster = n_clusters if two_step_clustering == False else n_cluster_2
    prod_onehot = np.zeros((df['order'].nunique(), final_n_cluster + feature_dim[3] + feature_dim[4] + feature_dim[5]))

    tmp_features = ['color', 'size', 'group'] if two_step_clustering == False else ['customer', 'color', 'size', 'group']
    for feature in tmp_features:
        for i, fes in enumerate(final_df.groupby('order')[feature].unique()):
            prod_onehot[i, fes] = 1
    
    pca = PCA(n_components=dim_pca)
    prod_pca_output = pca.fit_transform(prod_onehot)

    customer_onehot = np.zeros((df['order'].nunique(), final_n_cluster + feature_dim[3] + feature_dim[4] + feature_dim[5]))
    
    tmp_features = ['color', 'size', 'group'] if two_step_clustering == False else ['product', 'color', 'size', 'group']
    for feature in tmp_features:
        for i, fes in enumerate(final_df.groupby('order')[feature].unique()):
            customer_onehot[i, fes] = 1

    pca = PCA(n_components=dim_pca)
    cust_pca_output = pca.fit_transform(customer_onehot)

    output = np.concatenate([output, prod_pca_output, customer_onehot], axis=1)

    # np.save('task1_one_hot.npy', output)

    output = sp.csr_matrix(output)
    print(output.shape)

    import pickle
    with open(f"etail/ours/clustering_onehot_pca_{memo}.pickle", 'wb') as f:
        pickle.dump(output, f)
    

import os
os.makedirs('./tmp', exist_ok=True)
final_df.to_csv(f'./tmp/task2_data_clustered_{memo}.csv', index=False)
breakpoint()
