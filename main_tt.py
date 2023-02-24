import torch
from utils import *
from torch import nn
from torch.nn import Linear
from torch_geometric.data import DataLoader
from model import *
from GCN import *
import torch.nn.functional as F
from sklearn.metrics import log_loss
from math import log
import argparse
import sys
from utils import seed_everything
from typing import Tuple, Optional
from torch import Tensor
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from DGCN import *
#import nni
import os
import random
from typing import List
import numpy
#from utils_mp import Subgraph

class Distillsub:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def train_and_evaluate(self, model, train_loader, test_loader, optimizer, args):
        model.train()
        accs, aucs, macros = [], [], []
        #epoch_num = self.get_epoch_num(args, is_tuning)
        criterion = torch.nn.CrossEntropyLoss()
        #criterion=log_loss()
        loss_ce=torch.nn.MSELoss()
        #criterion_div = DistillKL(args.kd_T)
        #path='./data/subgraph/'+args.dataset_name
        for i in range(args.epoch_num):
            loss_all = 0
            for data in train_loader:
                optimizer.zero_grad()
                y_t,_=model( data.x, data.edge_index, data.batch)
                data.y=torch.LongTensor(data.y.numpy())
                loss_b = criterion(y_t,data.y)  # Computing the loss.
                loss=loss_b
                loss.backward()
                optimizer.step()
                loss_all += loss.item()
            epoch_loss = loss_all / len(train_loader.dataset)
            train_micro, train_auc, train_macro = self.eval(model,train_loader,args)
            title = "Start Train" 
            print(f'({title}) | Epoch={i:03d}, loss={epoch_loss:.4f}, \n'
                  f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                  f'train_auc={(train_auc * 100):.2f}')
            if (i + 1) % args.test_interval == 0:
                test_micro, test_auc, test_macro = self.eval(model, test_loader, args)
                accs.append(test_micro)
                aucs.append(test_auc)
                macros.append(test_macro)
                text = f'({title} Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                       f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
                print(text)
                with open(args.save_result, "a") as f:
                    f.writelines(text)
        torch.save(model,'GCN_pd.pkl')
        accs, aucs, macros = numpy.sort(numpy.array(accs)), numpy.sort(numpy.array(aucs)), \
                             numpy.sort(numpy.array(macros))
        return accs.max(), aucs.max(), macros.max()

    def eval(self, model, loader, args):
        model.eval()
        preds, trues, preds_prob = [], [], []
        for data in loader:
            #data = data.to(self.device)
            c,_=model(data.x, data.edge_index, data.batch)
            pred = c.max(dim=1)[1]
            preds += pred.detach().cpu().tolist()
            preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
            trues += data.y.detach().cpu().tolist()
        fpr, tpr, _ = metrics.roc_curve(trues, preds_prob)
        train_auc = metrics.auc(fpr, tpr)
        if numpy.isnan(train_auc):
            train_auc = 0.5
        train_micro = f1_score(trues, preds, average='micro')
        train_macro = f1_score(trues, preds, average='macro', labels=[0, 1])
        
        return train_micro, train_auc, train_macro

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--epoch_num', type=int, default=20)
        parser.add_argument('--seed', type=int, default=112078)
        parser.add_argument('--k_hop', type=int, default=1)
        parser.add_argument('--sub_size', type=int, default=15)
        parser.add_argument('--dataset_name', type=str, default="COBRE")
        parser.add_argument('--node_features', type=str,
                            choices=['degree', 'LDP', 'eigen', 'adj'],
                            # LDP is degree profile and adj is edge profile
                            default='degree')
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--k_fold_splits', type=int, default=5)
        parser.add_argument('--enable_nni', action='store_true')
        parser.add_argument('--dropout', type=float, default=0.5)
        #parser.add_argument('--no_vis', action='store_true')
        parser.add_argument('--top_k', type=int, default=10)
        #parser.add_argument('--repeat', type=int, default=1)
        parser.add_argument('--kd_T', type=int, default=4)
        parser.add_argument('--test_interval', type=int, default=5)
        parser.add_argument('--save_result', type=str, default='test_ex1')
        args = parser.parse_args()
        #if os.path.exists(args.save_result):
        #    os.remove(args.save_result)
        # load datasets
        adj, feature, y=load_dataset(args)
        adj = torch.FloatTensor(adj)
        feature=torch.FloatTensor(feature) 
        #print(feature.shape,'^^^^^^&&&&&&&&&&&&&&&')
        dataset=create_sub(feature,adj,y)
        #sub_dataset=extract_subgraphs(args.top_k,dataset)
        # init model
        seed_everything(random.randint(1, 1000000))  # use random seed for each run
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        accs, aucs, macros= [], [], []
        seed_everything(random.randint(1, 1000000))  # use random seed for each run
        skf = StratifiedKFold(n_splits=args.k_fold_splits, shuffle=True)
        for train_index, test_index in skf.split(dataset, y):
            model = GCN(128)
            #model = GAT(116,128,128)
            #model = GIN(args.hidden_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            train_set = [dataset[i] for i in train_index]
            test_set = [dataset[i] for i in test_index]
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
            test_micro, test_auc, test_macro = self.train_and_evaluate(model, train_loader, test_loader,
                                                                           optimizer, args)
            print(f'(Initial Performance Last Epoch) | test_micro={(test_micro * 100):.2f}, '
                      f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}')
            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)
        mean_acc = np.mean(np.array(accs))
        std_acc = np.std(np.array(accs))
        mean_auc = np.mean(np.array(aucs))
        std_auc = np.std(np.array(aucs))
        mean_mac = np.mean(np.array(macros))
        std_mac = np.std(np.array(macros))
        print("Mean Acc:", mean_acc,'±',std_acc)
        print("Mean Auc:", mean_auc,'±',std_auc)
        print("Mean Mac:", mean_mac,'±',std_mac)

if __name__ == '__main__':
    Distillsub().main()

