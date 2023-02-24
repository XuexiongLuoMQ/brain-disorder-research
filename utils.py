import numpy as np
import torch
from torch_geometric.data import Data, Batch
from scipy.sparse import coo_matrix
import scipy.io as scio
import torch_geometric.utils as utils
import random
#from __future__ import print_function
import os
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Union
import numpy
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch.utils.data import random_split
import numpy as np
# newly added dependency for tensor decomposition
import tensorly as tl
from tensorly.decomposition import parafac, tucker  # cp decomposition and tucker decomposition
from scipy.io import savemat
from numpy import linalg as LA

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def seed_everything(seed):
    print(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # set random seed for numpy
    # set deterministic for conv in cudnn
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True 
    torch.manual_seed(seed) # set random seed for CPU
    torch.cuda.manual_seed_all(seed) # set random seed for all GPUs
    
def binning(a, n_bins=10):
    n_graphs = a.shape[0]
    n_nodes = a.shape[1]
    _, bins = np.histogram(a, n_bins)
    binned = np.digitize(a, bins)
    binned = binned.reshape(-1, 1)
    enc = OneHotEncoder()
    return enc.fit_transform(binned).toarray().reshape(n_graphs, n_nodes, -1).astype(np.float32)

def get_data(dataset = "Training", feature = False):
    healthy = []
    patient = []
    if dataset == "Training" and feature == False:
        for i in range(1,11):
            healthy.append([np.genfromtxt('../'+dataset+'/Health/sub'+str(i)+'/common_fiber_matrix.txt'), 1])
            patient.append([np.genfromtxt('../'+dataset+'/Patient/sub'+str(i)+'/common_fiber_matrix.txt'), 0])
    
    elif dataset == "Testing" and feature == False:
        for i in range(1,6):
            healthy.append([np.genfromtxt('../'+dataset+'/Health/sub'+str(i)+'/common_fiber_matrix.txt'), 1])
            patient.append([np.genfromtxt('../'+dataset+'/Patient/sub'+str(i)+'/common_fiber_matrix.txt'), 0])
    
    elif dataset == "Training" and feature == True:
        for i in range(1,11):
            healthy.append([np.genfromtxt('../'+dataset+'/Health/sub'+str(i)+'/pcc_fmri_feature_matrix_0.txt'), 1])
            patient.append([np.genfromtxt('../'+dataset+'/Patient/sub'+str(i)+'/pcc_fmri_feature_matrix_0.txt'), 0])
    
    elif dataset == "Testing" and feature == True:
        for i in range(1,6):
            healthy.append([np.genfromtxt('../'+dataset+'/Health/sub'+str(i)+'/pcc_fmri_feature_matrix_0.txt'), 1])
            patient.append([np.genfromtxt('../'+dataset+'/Patient/sub'+str(i)+'/pcc_fmri_feature_matrix_0.txt'), 0])       
    data = []
    for i in range(len(healthy)):
        data.append(healthy[i])
        data.append(patient[i])
    del healthy, patient
    
    return data

# Using the PyTorch Geometric's Data class to load the data into the Data class needed to create the dataset

def load_dataset(args):
    if args.dataset_name == 'COBRE':
                dataFile = './data/COBRE/adj_cr.mat'
                data = scio.loadmat(dataFile)
                data=(data['correlation_matrix'])
                #print(data,'PPPPPPPPPPPP')
                data_f=refine_matrix(data)  
                #feature=define_feature(data_f,data)
                feature=data
                #feature=compute_x(data_f, args)
                #for n in range(feature.shape[0]):
                 #   np.savetxt('./data/adhd_degree/adhd{}.txt'.format(n),feature[n])
                dataFile = './data/COBRE/label.mat'
                data = scio.loadmat(dataFile)
                y_lab=(data['label222'])
                label=get_label(y_lab)
    
                return data_f, feature, label
    elif args.dataset_name == 'ParkD':
                dataFile = './data/ParkD/adj_ne1.mat'
                data = scio.loadmat(dataFile)
                data=(data['correlation_matrix'])
                data_f=refine_matrix(data)  
                #feature=define_feature(data_f,data)
                feature=data
                #feature=compute_x(data_f, args)
                #print(feature[0].shape,'LLLLLLLLLLL')    
                dataFile = './data/ParkD/label.mat'
                data = scio.loadmat(dataFile)
                y_lab=(data['label11'])
                label=get_label(y_lab)

                return data_f, feature, label
    #self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx

def create_dataset(feat,adj,label):
    # dataset_list = []
    x_list = []
    adj_list = []
    edge_index_list =[]
    y_list = []
    for i in range(len(adj)):      
        edge_index_coo = coo_matrix(adj[i])
        edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype = torch.long)
        # graph_data = Data(x = feat[i], adj=adj[i], edge_index=edge_index_coo, y = torch.tensor(label[i]))
        # dataset_list.append(graph_data)
        x_list.append(feat[i])
        adj_list.append(adj[i])
        edge_index_list.append(edge_index_coo)
        y_list.append(torch.tensor(label[i]))
    return torch.stack(x_list), torch.stack(adj_list), torch.stack(edge_index_list), torch.stack(y_list) 

def create_sub(feat,adj,label):
    dataset_list = []
    for i in range(len(adj)):    
        edge_index_coo = coo_matrix(adj[i])
        edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype = torch.long)
        graph_data = Data(x = feat[i], adj=adj[i], edge_index=edge_index_coo, y = torch.tensor(label[i]))
        dataset_list.append(graph_data)
    return dataset_list
    
def adj_process(adj):
    g_num, n_num, n_num = adj.shape
        #adjs = adjs.detach()
    for i in range(g_num):
        adj[i] = adj[i] + torch.eye(n_num)
        adj[i][adj[i]>0.] = 1.
        degree_matrix = torch.sum(adj[i], dim=-1, keepdim=False)
        degree_matrix = torch.pow(degree_matrix,-1)
        degree_matrix[degree_matrix == float("inf")] = 0.
        degree_matrix = torch.diag(degree_matrix)
        adj[i] = torch.mm(degree_matrix, adj[i])
    return adj
def get_label(label):
    for i in range(len(label)):
        if label[i]==[1]:
            label[i]=1
        elif label[i]==[-1]:
            label[i]=0  
    return label

def refine_matrix(data):
    N,V,V=data.shape
    for n in range(N):
        for v in range(V):
            for i in range(V):
                if data[n][v][i]>0.8:
                    data[n][v][i]=1
                else:
                    data[n][v][i]=0
    return data

def define_feature(adj, W):
    feature=[]
    for i in range(len(adj)):
        #degree_matrix = np.count_nonzero(adj[i], axis=1).reshape(116,1)
        #weight_matrix = np.diag(np.sum(W[i], axis=1)).diagonal().reshape(116,1)
        #feature_matrix = np.hstack((degree_matrix, weight_matrix))
        #feature.append(np.array(feature_matrix))
        degs = np.sum(np.array(adj[i]), 1)
        degs = np.expand_dims(degs, axis=1)
        feature.append(np.array(degs)) 
    return feature

def compute_x(a1, args):
    a1=torch.Tensor(a1)
    # construct node features X
    if args.node_features == 'identity':
        x = torch.cat([torch.diag(torch.ones(a1.shape[1]))] * a1.shape[0]).reshape([a1.shape[0], a1.shape[1], -1])
        x1 = x.clone()
        
    elif args.node_features == 'degree':
        a1b = (a1 != 0).float()
        x1 = a1b.sum(dim=2, keepdim=True)

    elif args.node_features == 'degree_bin':
        a1b = (a1 != 0).float()
        x1 = binning(a1b.sum(dim=2))

    elif args.node_features == 'adj': # edge profile
        x1 = a1.float()
        
    elif args.node_features == 'LDP': # degree profile
        a1b = (a1 != 0).float()
        x1 = []
        n_graphs: int = a1.shape[0]
        for i in range(n_graphs):
            x1.append(LDP(nx.from_numpy_array(a1b[i].numpy())))

    elif args.node_features == 'eigen':
        _, x1 = LA.eig(a1.numpy())
    x1 = torch.Tensor(x1).float()
    
    return x1
    
def LDP(g, key='deg'):
    x = np.zeros([len(g.nodes()), 5])
    deg_dict = dict(nx.degree(g))
    for n in g.nodes():
        g.nodes[n][key] = deg_dict[n]
    for i in g.nodes():
        nodes = g[i].keys()
        nbrs_deg = [g.nodes[j][key] for j in nodes]
        if len(nbrs_deg) != 0:
            x[i] = [
                np.mean(nbrs_deg),
                np.min(nbrs_deg),
                np.max(nbrs_deg),
                np.std(nbrs_deg),
                np.sum(nbrs_deg)
            ]
    return x
def extract_subgraphs(k,data):
    print("Extracting {}-hop subgraphs...".format(k))
        # indicate which node in a graph it is; for each graph, the
        # indices will range from (0, num_nodes). PyTorch will then
        # increment this according to the batch size
    subgraph =[]
        # Each graph will become a block diagonal adjacency matrix of
        # all the k-hop subgraphs centered around each node. The edge
        # indices get augumented within a given graph to make this
        # happen (and later are augmented for proper batching)
    #subgraph_edge = []
        # This identifies which indices correspond to which subgraph
        # (i.e. which node in a graph)
    #subgraph_indicator = []
    #subgraph_feat=[]
        # This gets the edge attributes for the new indices
        #if self.use_subgraph_edge_attr:
            #self.subgraph_edge_attr = []
    #for i in range(len(self.dataset)):
            #if self.cache_path is not None:
                #filepath = "{}_{}.pt".format(self.cache_path, i)
                #if os.path.exists(filepath):
                    #continue
        #edge_index_coo = coo_matrix(graphs[i])
        #edge_index= torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype = torch.long)     
    graph = data
    edge_indices = []
    #edge_attributes = []
    indicators = []
    edge_index_start = 0
        #adj_start=0
        #print(graph.num_nodes,'VVVVVVVVVVVVVVVVV')
    def adjust_x(idx):
        #Generate node features for subgraphs
        return graph.x[idx]        
    #print(graph.num_nodes,'CCCCCCCCCCCCCCCC')
    for node_idx in range(graph.num_nodes):
        sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                    node_idx, 
                    k, 
                    graph.edge_index,
                    relabel_nodes=True, 
                    num_nodes=graph.num_nodes
                    )
        #print(sub_nodes.shape,'1111111111111')
            #print(sub_nodes,'11111;;;;;;;;;;;;;;;;;;;;;11111111')
            #print(sub_edge_index,'22222222222222')
        #subX=torch.zeros([sub_nodes.shape[0],graph.x.shape[1]])
            #sub_a=torch.zeros([sub_nodes.shape[0],sub_nodes.shape[0]])
        #for i in range(sub_nodes.shape[0]):
        subX=adjust_x(sub_nodes)
        #print(subX.shape,'444444444444444444444')
        sub_adj = utils.to_scipy_sparse_matrix(sub_edge_index)
        sub_adj= sub_adj.toarray()
        sub_adj=torch.tensor(sub_adj)
            #print(sub_adj.shape,'12346')
            #for i in range(sub_nodes.shape[0]):
            #    sub_a[i]=sub_adj[i]
            #print(np.array(sub_a),'MMMMMMMMMMMMM')
            #print(sub_a.shape,'LLLLLLLLLLLLLL')
        edge_indices= sub_edge_index + edge_index_start   
        #indicators=torch.zeros(sub_nodes.shape[0]).fill_(node_idx)
        subgraph.append(Data(node =sub_nodes , edge=edge_indices, subX=subX,adj=sub_adj))
    #for node_idx in range(graph.num_nodes):
        #sub.append(subgraph[node_idx])
    batch = Batch().from_data_list(subgraph)

    return batch
