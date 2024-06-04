# E-HyperGCN+ : E-commerce Return Prediction on HyperGraph based Graph Convolutional Networks with Clustering

[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://nips.cc/) [![Paper](http://img.shields.io/badge/paper-arxiv.1809.02589-B31B1B.svg)](https://arxiv.org/abs/1809.02589) 

This code is motivated by Source code for [NeurIPS 2019](https://nips.cc/) paper: [**HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs**](https://papers.nips.cc/paper/8430-hypergcn-a-new-method-for-training-graph-convolutional-networks-on-hypergraphs)

![](./hmlap.png)

**Overview of E-HyperGCN+:** * In the e-commerce market, predicting product returns is crucial for companies selling products, as it can significantly impact their profitability. Especially when returns occur, customers may opt to return individual products from their order or return the entire order altogether. To address this scenario, we design the problem into two stages: return prediction at the order level and the product level within the order. For each level of prediction, we propose an efficient hypergraph-based algorithm called {\em E-HyperGCN+}, which allows customers to organize products and orders effectively. Furthermore, we introduce a method incorporating Graph Convolutional Networks (GCN), the most prominent methodology for understanding graph representations, into hypergraph. Additionally, we utilize multi-hot encoding-based K-mean clustering to design feature vectors for individual nodes in the hypergraph, aiming to create a hypergraph with high-quality embedding features. *

<!-- ### Dependencies

- Compatible with PyTorch 1.0 and Python 3.x.
- For data (and/or splits) not used in the paper, please consider tuning hyperparameters such as [hidden size](https://github.com/malllabiisc/HyperGCN/blob/master/model/networks.py#L25), [learning rate](https://github.com/malllabiisc/HyperGCN/blob/master/config/config.py#L49), [seed](https://github.com/malllabiisc/HyperGCN/blob/master/config/config.py#L28), etc. on validation data. -->

### How to Run the Code? (Node classifiction):

#### Task 1

1) Run task1.ipynb ; Create label & hypergraph etc. 
2) Run task1_onehot.py  ; Create clustering feature
3) To start training model run:

```shell
python hypergcn.py --mediators True --split 1 --data etail --dataset ours --features <name of feature, ex; clustering_onehot > --rate 0.05 --result task1_result --gpu 0 --fast True --epoch 1000 --task 1
```



#### Task 2

1) Run task2_preprocessing.py : Convert data from product-level prediction(task2) like order-level prediction(task1).
2) Run task2.ipynb : Create label & hypergraph etc.
3) Run task2_onehot.py ; Create clustering feature
4) To start training model run:

```shell
python hypergcn.py --mediators True --split 1 --data etail --dataset ours2 --features <name of feature, ex; cluster_onehot > --rate 0.05 --result clus_0.05_task2 --gpu 3 --fast True --epoch 1000 --task 2
```

5) To start task 2 prediction run:

```shell
python task2_prediction.py --mediators True --split 1 --data etail --dataset ours2 --features <name of feature, ex; cluster_onehot > --rate 0.05 --result clus_0.05_task2 --gpu 3 --fast True --epoch 300 --task 2
```


#### Some Minor Code for Reproduction Results in Paper
```shell
CUDA_VISIBLE_DEVICES=1 python hypergcn.py --mediators True --split 1 --data etail --dataset ours --features clustering_onehot_pca --rate 0.03 --result clus_pca_0.03_task1 --gpu 3 --fast True --epoch 5000 --task 1

CUDA_VISIBLE_DEVICES=1 python hypergcn.py --mediators True --split 1 --data etail --dataset ours2 --features clustering_onehot_pca --rate 0.03 --result clus_pca_0.03_task2 --gpu 3 --fast True --epoch 5000 --task 2

 python task2_prediction.py --mediators True --split 1 --data etail --dataset ours2 --features clustering_onehot_pca --rate 0.03 --result clus_pca_0.03_task2 --gpu 3 --fast True --epoch 5000 --task 2
 ```


### Citation:

```bibtex
@incollection{hypergcn_neurips19,
title = {HyperGCN: A New Method For Training Graph Convolutional Networks on Hypergraphs},
author = {Yadati, Naganand and Nimishakavi, Madhav and Yadav, Prateek and Nitin, Vikram and Louis, Anand and Talukdar, Partha},
booktitle = {Advances in Neural Information Processing Systems (NeurIPS) 32},
pages = {1509--1520},
year = {2019},
publisher = {Curran Associates, Inc.}
}

```
