import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool 
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        self.drop_out = Dropout(0.5)
        self.conv1 = GCNConv(116, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 2)
    def forward(self, x,edge_index,batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.01, training=self.training)
        x=self.drop_out(x)
        x_t = self.lin1(x)
        return x_t, x
    
class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=8, concat=False)
        self.conv2 = GATConv(h_feats, out_feats, heads=8, concat=False)
        self.lin1 = Linear(out_feats, 2)
        self.drop_out = Dropout(0.5)
    def forward(self, x,edge_index,batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x) 
        x = global_mean_pool(x, batch)
        #x = F.dropout(x, training=self.training)
        x=self.drop_out(x)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = F.softmax(x, dim=1)
        x_t = self.lin1(x)
        return x_t, x

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
        #self.conv1 = GINConv(116,  h_feats)
        #self.conv2 = GATConv(h_feats, out_feats, heads=8, concat=False)
        self.lin1 = Linear( hidden, 2)
        self.drop_out = Dropout(0.5)
    def forward(self, x,edge_index,batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)   
        x = global_mean_pool(x, batch)
        #x = F.dropout(x, training=self.training)
        x=self.drop_out(x)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = F.softmax(x, dim=1)
        x_t = self.lin1(x)
        return x_t, x