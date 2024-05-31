import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sparse
from scipy.stats import truncnorm 
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

from config import config
args = config.parse()
import pdb

# seed
import os, torch, numpy as np
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# gpu, seed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)


# load data
from data import data
dataset, train, valid, test = data.load(args)
print("length of train is", len(train))
print("length of valid is", len(valid))
print("length of test is", len(test))


# # initialise HyperGCN
from model import model
HyperGCN = model.initialise(dataset, args)

print("============= best model inference =============")
# load best model
HyperGCN['model'].load_state_dict(torch.load(f'results/{args.result}-best-model.pth'))
acc = model.valid(HyperGCN, dataset, valid, args)
print("accuracy:", float(acc), ", error:", float(100*(1-acc)))


print("============= load text query predictions =============")
HyperGCN['model'].load_state_dict(torch.load(f'results/{args.result}-best-model.pth'))
basket_pred = model.test(HyperGCN, dataset, train + valid + test, args)

# breakpoint()

# file directory
train_data_dir = "data/task2_train_label.txt"
val_data_dir = "data/task2_valid_label.txt"
test_data_dir = "data/task2_test_query.txt"

task2_basket_data = "data/task2_basket_data.txt"
# TODO : put path of task2_basket_test_query with prediction label 
task2_basket_test_query_result = "data/task2_basket_test_query_result.txt"

# TODO : need to change

hypergraph_path = "data/etail/ours/hypergraph.pickle"
with open(hypergraph_path, 'rb') as file:
    hypergraph = pickle.load(file)

f_train_bsk_lb = open(task2_basket_data, "r")
line_train_bsk_lb = f_train_bsk_lb.readlines()

f_test_bsk_lb = open(task2_basket_test_query_result, "r")
line_test_bsk_lb = f_test_bsk_lb.readlines()

total_basket_lb = line_train_bsk_lb + line_test_bsk_lb

basket = dict()

for l in total_basket_lb:
    info = l.strip().split("\t")
    order, returns = info[0], info[1]
    basket[int(order)] = int(returns)

# # TODO : complete the bsk_return_pred
# def bsk_return_pred(model_save_path, order):
#     model = ???? # TODO prediction 불러오기
#     bsk_return = ??? # max(0에 대해 모델이 뱉은 pred, 1 =, 2=)
#     return bsk_return
    
def bsk_return_pred(order):
    prob = basket_pred[int(order)][1]

#     return prob[2].item()
# AUC Score: 0.524217290561405
# accuracy: 0.48533530029504424

#     return prob[1].item()
# AUC Score: 0.5606571876950731
# accuracy: 0.4880357053558034

#     return prob[2].item() + (0.5 * prob[1].item())
# AUC Score: 0.5615265855585243
# accuracy: 0.48555783367505123

    return prob[2].item() + prob[1].item()
# AUC Score: 0.562575022417739
# accuracy: 0.5073936090413562

    # return (0.5 * prob[2].item()) + prob[1].item()

def prd_return_pred(product):
    order_list = hypergraph[int(product)]
    total_bsk = len(order_list)
    return_bsk = 0
    
    for order in order_list:
        if basket[int(order)] == 2:
            return_bsk += 1
        elif basket[int(order)] == 1:
            return_bsk += bsk_return_pred(int(order)) #TODO
        else:
            pass
            
    prd_return = return_bsk / total_bsk
        
    return prd_return
    
print("================== validation restuls ================")
f = open(val_data_dir, "r")
line = f.readlines()

y_test = []
y_pred_prob = []
total_sample = 0
correct_sample = 0
for l in line:
    info = l.strip().split("\t")
    order, product, return_gt = info[0], info[1], info[2]
    
    bsk_return_prob = bsk_return_pred(int(order))
    prod_return_prob = prd_return_pred(int(product))

    pred_1 = bsk_return_prob * prod_return_prob
    pred_0 = (1-bsk_return_prob) * (1-prod_return_prob)

    preds = [pred_0, pred_1]
    preds = F.softmax(torch.tensor(preds), dim=0)

    y_test.append(int(return_gt))
    y_pred_prob.append(preds[1].item()) # TODO : ??

    if preds[1].item() >= 0.5:
        pred_label = 1
    else:
        pred_label = 0
    
    if int(return_gt) == pred_label:
        correct_sample += 1
    total_sample += 1

# AUC 점수 계산
auc_score = roc_auc_score(y_test, y_pred_prob)
accuary = correct_sample / total_sample
print(f"AUC Score: {auc_score}")
print(f"accuracy: {accuary}")




# start predict label
print("================== save test prediction ================")
f = open(test_data_dir, "r")
line = f.readlines()

task2_result = open("data/task2_test_query_result.txt", "w")

for l in line:
    info = l.strip().split("\t")
    order, product = info[0], info[1]
    pred = bsk_return_pred(int(order)) * prd_return_pred(int(product))
    if pred >= 0.5:
        task2_result.write(f"{order}\t{product}\t{1}\n")
    else:
        task2_result.write(f"{order}\t{product}\t{0}\n")