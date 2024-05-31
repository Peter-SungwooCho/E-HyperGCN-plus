# parse arguments ([ConfigArgParse](https://github.com/bw2/ConfigArgParse))
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

# breakpoint()

# train and test HyperGCN
HyperGCN = model.train(HyperGCN, dataset, train, valid, test, args)

f= open(f'results/{args.result}-vaild_acc.txt',"a")
# torch.save(HyperGCN['model'].state_dict(), f'results/{args.result}-last model.pth')

print("============= best model inference =============")
f.write("============= best model inference =============")
# load best model
HyperGCN['model'].load_state_dict(torch.load(f'results/{args.result}-best-model.pth'))
acc = model.valid(HyperGCN, dataset, valid, args)
print("accuracy:", float(acc), ", error:", float(100*(1-acc)))
f.write(f"accuracy: {float(acc)}, error: {float(100*(1-acc))}")



print("============= save text query predictions =============")
f.write("============= save text query predictions =============")
HyperGCN['model'].load_state_dict(torch.load(f'results/{args.result}-best-model.pth'))
pred = model.test(HyperGCN, dataset, test, args)

if args.task == 1:
    query = open("data/task1_test_query.txt", "r")
    line = query.readlines()

    result = open("data/task1_test_query_result.txt", "w")
    for l in line:
        order = int(l.strip())
        pred_label = pred[order][0]
        result.write(f"{order}\t{pred_label}\n")

    result.close()
    query.close()

    print(f'saved : data/task1_test_query_result.txt')
    f.write(f'saved : data/task1_test_query_result.txt')

    f.close()
else:
    query = open("data/task2_basket_test_query.txt", "r")
    line = query.readlines()

    result = open("data/task2_basket_test_query_result.txt", "w")
    for l in line:
        order = int(l.strip())
        pred_label = pred[order][0]
        result.write(f"{order}\t{pred_label}\n")

    result.close()
    query.close()

    print(f'saved : data/task2_basket_test_query_result.txt')
    f.write(f'saved : data/task2_basket_test_query_result.txt')

    f.close()