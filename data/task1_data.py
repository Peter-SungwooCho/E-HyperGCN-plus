import numpy as np
from scipy.sparse import csr_matrix
import pickle 

print("start - customer")
# 희소 행렬을 생성하기 위한 리스트 초기화
rows = []
cols = []
data = []

with open("../data/task1_data.txt", "r") as f:
    f.readline()
    while True:
        line = f.readline()
        if not line:  # 파일 읽기가 종료된 경우
            break
        order, product, customer, color, size, group = line.strip().split(',')
        order, product, customer, color, size, group = int(order), int(product), int(customer), int(color), int(size), int(group)
        
        rows.append(order)
        cols.append(customer)
        data.append(1)  # BoW에서는 단순히 출현 횟수를 카운트하므로 1을 더해줌

# 희소 행렬 생성
num_orders = 849185
num_customers = 342039
customers_bow_sparse = csr_matrix((data, (rows, cols)), shape=(num_orders, num_customers))

print("svd - customers")
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100)
customers_bow_sparse_reduced = svd.fit_transform(customers_bow_sparse)

print("binarizer - customers")
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0)
customers_bow_sparse_reduced_binarized = binarizer.fit_transform(customers_bow_sparse_reduced)

print("start - product")
rows = []
cols = []
data = []

with open("../data/task1_data.txt", "r") as f:
    f.readline()
    while True:
        line = f.readline()
        if not line:  # 파일 읽기가 종료된 경우
            break
        order, product, customer, color, size, group = line.strip().split(',')
        order, product, customer, color, size, group = int(order), int(product), int(customer), int(color), int(size), int(group)
        
        rows.append(order)
        cols.append(product)
        data.append(1)  # BoW에서는 단순히 출현 횟수를 카운트하므로 1을 더해줌

# 희소 행렬 생성
num_orders = 849185
num_product = 58415
product_bow_sparse = csr_matrix((data, (rows, cols)), shape=(num_orders, num_product))

print("svd - product")
svd = TruncatedSVD(n_components=50)
product_bow_sparse_reduced = svd.fit_transform(product_bow_sparse)

print("binarizer - product")
binarizer = Binarizer(threshold=0.0)
product_bow_sparse_reduced_binarized = binarizer.fit_transform(product_bow_sparse_reduced)

import scipy.sparse as sp
matrix = sp.csr_matrix(np.hstack([customers_bow_sparse_reduced_binarized, product_bow_sparse_reduced_binarized]))
with open("etail/ours/bow_svd_cp.pickle", 'wb') as f:
    pickle.dump(matrix, f)