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
dataset, train, test = data.load(args)
print("length of train is", len(train))



# # initialise HyperGCN
from model import model
HyperGCN = model.initialise(dataset, args)

# breakpoint()

# train and test HyperGCN
HyperGCN = model.train(HyperGCN, dataset, train, test, args)

torch.save(HyperGCN['model'].state_dict(), f'{args.result}-last model.pth')

acc = model.test(HyperGCN, dataset, test, args)
print("accuracy:", float(acc), ", error:", float(100*(1-acc)))

f= open(f'{args.result}-vaild_acc.txt',"a")
f.write(f"\naccuracy: {float(acc)} , error: {float(100*(1-acc))}")
f.close()