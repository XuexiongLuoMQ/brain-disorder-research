import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool 
import torch.nn as nn
import numpy as np
import copy 
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
class GCN(nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(116, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #self.lin1 = Linear(hidden_channels, 2)

    def forward(self, x, adj):
        # 1. Obtain node embeddings 
        x = self.conv1(x, adj)
        x = F.relu(x)
        #x = self.conv2(x, adj)
        #x = F.relu(x)
        # 2. Readout layer
        #x_graph = global_mean_pool(x1, data.batch)  # [batch_size, hidden_channels]
        #x_sub=torch.sum(x1, dim=1)
        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.01, training=self.training)
        #x = self.lin1(x)    
        return x

class GAT(torch.nn.Module):
    def __init__(self, h_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(116, h_feats, heads=8, concat=False)
        #self.drop_out = Dropout(0.5)
    def forward(self, x,edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)    
        #x = global_mean_pool(x, batch)
        #x = F.dropout(x, training=self.training)
        #x=self.drop_out(x)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = F.softmax(x, dim=1)
        #x_t = self.lin1(x)
        return  x
    
class GIN(torch.nn.Module):
    def __init__(self,  hidden):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(116, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
    def forward(self, x,edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = global_mean_pool(x, batch)
        #x = F.dropout(x, training=self.training)
        #x=self.drop_out(x)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = F.softmax(x, dim=1)
        #x_t = self.lin1(x)
        return  x

    
    
class En(nn.Module):
    def __init__(self, hidden_channels):
        super(En, self).__init__()
        self.gc1 = nn.Linear(1, hidden_channels, bias=False)
        #self.gc2 = nn.Linear(hiddendim*2, hiddendim*2, bias=False)
        #self.gc3 = nn.Linear(hiddendim*2, hiddendim, bias=False)
        self.gc4 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        #self.proj_head = nn.Sequential(nn.Linear(outputdim, outputdim), nn.ReLU(inplace=True), nn.Linear(outputdim, outputdim))
        self.leaky_relu = nn.LeakyReLU(0.5)
        #self.dropout = nn.Dropout(dropout)
        #self.batch=batch
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(torch.matmul(adj, x)))
        #x=self.dropout(x)
        x =F.relu(self.gc4(torch.matmul(adj, x))) 
        #out, _ = torch.max(x, dim=1)
        #out = global_add_pool(x,self.batch)
        #out=self.proj_head(out)
        return x
    
class Net(nn.Module):
    def __init__(self, hidden,top_k):
        super(Net, self).__init__()
        self.k=top_k
        self.hidden=hidden
        self.GCN = GCN(hidden)
        #self.GAT = GAT(hidden)
        #self.GIN = GIN(hidden)
        self.lin0 = Linear(hidden, 1)
        self.lin1 = Linear(hidden, 2)
        self.lin2 = Linear(hidden, hidden)
        self.weight = nn.Parameter(torch.FloatTensor(hidden, hidden))
        self.F=nn.Tanh()
        
    def forward(self, sub, sub_data,h_t):
        sub_a=[]
        ss_bb=[]
        #print(sub,'1111111111111<<<<<<<<<<<<')
        #sub=torch.tensor(sub)
        #sub_e=self.GAT(sub_data.x,sub_data.edge_index)
        sub_e=self.GCN(sub_data.x,sub_data.edge_index)
        #sub_e=self.GIN(sub_data.x,sub_data.edge_index)
        #print(sub_e.shape,'ZZZZZZZZZZZZZZZZZZZZ')
        sub_e=global_mean_pool(sub_e, sub_data.batch)
        for t in range(sub_e.shape[0]):
            sub_i=self.F(torch.matmul(sub_e[t],self.weight))
            sub_i=torch.matmul(sub_i,torch.reshape(h_t,[-1,1]))
            sub_a.append(sub_i)
        sub_ht=torch.stack(sub_a)
        att=torch.softmax(sub_ht, dim=0)
        h_s=(att * sub_e).sum(0,keepdim=True)
        y_s=self.lin1(h_s)
        _, a_i = torch.topk(att.squeeze(-1),dim=0,k=15)
        for x in a_i:
            dd=sub[x]
            ss_bb.append(dd)    
        #print(ss_bb,'MMMMMMMMMMM')
        att_s = sub_e[a_i]
        h_sub=att_s.sum(0,keepdim=True)
        out=self.lin1(h_sub)
                        
        return y_s,att,a_i,ss_bb, h_sub,out
                                   
        
        
        
        
        
        
        
        
        
        
        
        
    

    
