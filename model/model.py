from model import networks
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F

from torch.autograd import Variable
from tqdm import tqdm
from model import utils
import torch.nn as nn

import pdb
def train(HyperGCN, dataset, T, v, t, args):
    """
    train for a certain number of epochs

    arguments:
	HyperGCN: a dictionary containing model details (gcn, optimiser)
	dataset: the entire dataset
	T: training indices
	args: arguments

	returns:
	the trained model
    """    
    
    hypergcn, optimiser = HyperGCN['model'], HyperGCN['optimiser']
    hypergcn.train()
    
    X, Y = dataset['features'], dataset['labels']

    max_acc = 0.0
    for epoch in tqdm(range(args.epochs)):

        optimiser.zero_grad()
        Z = hypergcn(X)
        loss = F.nll_loss(Z[T], Y[T])

        loss.backward()
        optimiser.step()

        if epoch % 10 == 0:
            acc = valid(HyperGCN, dataset, v, args)

            print("epoch:", epoch, "loss:", float(loss), "accuracy:", float(acc), ", error:", float(100*(1-acc)), ", best accuracy:", float(max_acc))

            if float(acc) > max_acc:
                max_acc = float(acc)
                torch.save(HyperGCN['model'].state_dict(), f'results/{args.result}-best-model.pth')
            # torch.save(HyperGCN['model'].state_dict(), f'{args.result}-{epoch}.pth')
            
            f= open(f'results/{args.result}-vaild_acc.txt',"a")
            f.write(f"\nepoch: {epoch}, loss: {float(loss)}, accuracy: {float(acc)} , error: {float(100*(1-acc))}")
            f.close()
            
    HyperGCN['model'] = hypergcn
    return HyperGCN



def valid(HyperGCN, dataset, v, args):
    """
    valid HyperGCN
    
    arguments:
	HyperGCN: a dictionary containing model details (gcn)
	dataset: the entire dataset
	t: test indices
	args: arguments

	returns:
	accuracy of predictions    
    """
    
    hypergcn = HyperGCN['model']
    hypergcn.eval()
    X, Y = dataset['features'], dataset['labels']
    
    Z = hypergcn(X) 
    return accuracy(Z[v], Y[v])
    

def test(HyperGCN, dataset, t, args):
    """
    test HyperGCN
    
    arguments:
	HyperGCN: a dictionary containing model details (gcn)
	dataset: the entire dataset
	t: test indices
	args: arguments

	returns:
	accuracy of predictions    
    """
    
    hypergcn = HyperGCN['model']
    hypergcn.eval()
    X, Y = dataset['features'], dataset['labels']
    
    Z = hypergcn(X) 

    pred_dict = {}
    for tid in t:
        pred_dict[tid] = (Z[tid].argmax().item(), torch.exp(Z[tid]))
        
    return pred_dict



def accuracy(Z, Y):
    """
    arguments:
    Z: predictions
    Y: ground truth labels

    returns: 
    accuracy
    """
    
    predictions = Z.max(1)[1].type_as(Y)
    correct = predictions.eq(Y).double()
    correct = correct.sum()

    accuracy = correct / len(Y)
    return accuracy



def initialise(dataset, args):
    """
    initialises GCN, optimiser, normalises graph, and features, and sets GPU number
    
    arguments:
    dataset: the entire dataset (with graph, features, labels as keys)
    args: arguments

    returns:
    a dictionary with model details (hypergcn, optimiser)    
    """
    
    HyperGCN = {}
    V, E = dataset['n'], dataset['hypergraph']
    Y = dataset['labels']
    
    if dataset['features'] is not None:
        X = dataset['features']
        
    else:
        learnable_emb = nn.Embedding(V, 2000)
        X = learnable_emb.weight
    
    print(f"rate : {args.rate}")
    # hypergcn and optimiser
    args.d, args.c = X.shape[1], Y.shape[1]
    hypergcn = networks.HyperGCN(V, E, X, args)
    optimiser = optim.Adam(list(hypergcn.parameters()), lr=args.rate, weight_decay=args.decay)


    # node features in sparse representation
    if args.features == 'learnable':
        X = sp.csr_matrix(normalise(np.array(X.detach().numpy())), dtype=np.float32)
        
        X_np = np.array(X.todense())
        X = torch.tensor(X_np, dtype=torch.float32, requires_grad=True)

        
    else:
        X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
        X = torch.FloatTensor(np.array(X.todense()))
    
    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

    # cuda
    args.Cuda = args.cuda and torch.cuda.is_available()
    if args.Cuda:
        hypergcn.cuda()
        X, Y = X.cuda(), Y.cuda()

    # update dataset with torch autograd variable
    dataset['features'] = Variable(X)
    dataset['labels'] = Variable(Y)

    # update model and optimiser
    HyperGCN['model'] = hypergcn
    HyperGCN['optimiser'] = optimiser
    return HyperGCN



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    epsilon = 1e-10 
    di = np.power(d+epsilon, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}

    return DI.dot(M)
