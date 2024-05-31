import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sparse
from scipy.stats import truncnorm 
import pickle
import pandas as pd


# file directory
tot_data_dir = "task2_data.txt"
train_data_dir = "task2_train_label.txt"
val_data_dir = "task2_valid_label.txt"
test_data_dir = "task2_test_query.txt"



# get file info
## train
total_dict = dict()

# train_dict = dict()
# val_dict = dict()

f_train = open(train_data_dir, "r")
line_train = f_train.readlines()

f_val = open(val_data_dir, "r")
line_val = f_val.readlines()

f_test = open(test_data_dir, "r")
line_test = f_test.readlines()

tot_line = line_train + line_val + line_test

for l in tot_line:
    info = l.strip().split("\t")
    if len(info) == 3:
        order, product, returns = info[0], info[1], info[2]
        total_dict[str(order)+"_"+str(product)] = int(returns)
    else:
        order, product = info[0], info[1]
        total_dict[str(order)+"_"+str(product)] = None

def check_list(lst):
    if 0 in lst and 1 in lst:
        return 1
    if None in lst:
        return None
    elif 0 in lst:
        return 0
    elif 1 in lst:
        return 2


df = pd.read_csv(tot_data_dir)
grouped = df.groupby('order')['product'].apply(list)

bsk_trainval_file  = open("task2_basket_data.txt", "w")
bsk_test_file = open("task2_basket_test_query.txt", "w")

for order, product in grouped.items():
    check = []
    for p in product:
        check.append(total_dict[str(order) + "_" + str(p)])
    label = check_list(check)
    if label == None:
        bsk_test_file.write(f"{order}\n")
    else:
        bsk_trainval_file.write(f"{order}\t{label}\n")
