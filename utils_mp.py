import os 
import torch
import numpy as np
from cytoolz import curry
import multiprocessing as mp
from scipy import sparse as sp
from sklearn.preprocessing import normalize, StandardScaler
from torch_geometric.data import Data, Batch
import torch_geometric.utils as utils

def standardize(feat, mask):
    scaler = StandardScaler()
    scaler.fit(feat[mask])
    new_feat = torch.FloatTensor(scaler.transform(feat))
    return new_feats
    
def preprocess(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.tensor(features)

class PPR:
    #Node-wise personalized pagerank
    def __init__(self, adj_mat, maxsize=200, n_order=2, alpha=0.85):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()
        
    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)
        
        idx = scores.argsort()[::-1][:self.maxsize]
        neighbor = np.array(x.indices[idx])
        
        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            neighbor = np.append(np.array([seed]), neighbor)
        else :
            seed_idx = seed_idx[0]
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]
            
        assert np.where(neighbor == seed)[0].size == 1
        assert np.where(neighbor == seed)[0][0] == 0
        
        return neighbor
    
    @curry
    def process(self, path, seed):
        ppr_path = os.path.join(path, 'ppr{}'.format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            #print ('Processing node {}.'.format(seed))
            neighbor = self.search(seed)
            torch.save(neighbor, ppr_path)
        else :
            print ('File of node {} exists.'.format(seed))
    
    def search_all(self, node_num, path):
        neighbor  = {}

        #print ("Extracting subgraphs")
        os.system('mkdir {}'.format(path))
        with mp.Pool() as pool:
            list(pool.imap_unordered(self.process(path), list(range(node_num)), chunksize=1000))
                
        #print ("Finish Extracting")
        for i in range(node_num):
            neighbor[i] = torch.load(os.path.join(path, 'ppr{}'.format(i)))
        torch.save(neighbor, path+'_neighbor')
        os.system('rm -r {}'.format(path))
        #print ("Finish Writing")
        
        return neighbor

class Subgraph:
    #Class for subgraph extraction
    
    def __init__(self,data, x, edge_index, path, maxsize=50, n_order=10):
        self.data=data
        self.x = x
        self.path = path
        self.edge_index = np.array(edge_index)
        self.edge_num = edge_index[0].size(0)
        self.node_num = x.size(0)
        #print(self.node_num,'CCCCCCCCCCCCCCC')
        self.maxsize = maxsize
        self.k_hop=n_order
        self.sp_adj = sp.csc_matrix((np.ones(self.edge_num), (edge_index[0], edge_index[1])), 
                                    shape=[self.node_num, self.node_num])
        self.ppr = PPR(self.sp_adj, n_order=n_order)
        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = []
        
    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)
            
    def adjust_edge(self, idx):
        #Generate edges for subgraphs
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i
            
        new_index = [[], []]
        nodes = set(idx)
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            edge = [dic[_] for _ in edge]
            #edge = [_ for _ in edge if _ > i]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
        return torch.LongTensor(new_index)

    def adjust_x(self, idx):
        #Generate node features for subgraphs
        return self.x[idx]            
    
    def build(self):
        #Extract subgraphs for all nodes 
        subx_list=[]
        subedge_list=[]
        self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            nodes = self.neighbor[i][:self.maxsize]
            #print(nodes,'11111111111111111')
            x = self.adjust_x(nodes)
            #print(x,'2222222222')
            edge = self.adjust_edge(nodes)
            subx_list.append(x)
            subedge_list.append(edge)
            #print(edge,'5555555555555555555')
            #self.subgraph.append(Data(x, edge))
        #batch = Batch().from_data_list(self.subgraph)
        return torch.stack(subx_list), torch.stack(subedge_list)

    def Distill(self):
        subX=[]
        sub_sub=[]
        # #Extract subgraphs for all nodes 
        # def adjust_x(idx):
        # #Generate node features for subgraphs
        #     return self.x[idx] 
        #print(self.data.num_nodes,'CCCCCCCCCCCCCC')       
        for node_idx in range(self.data.num_nodes):
            sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                    node_idx, 
                    self.k_hop, 
                    self.data.edge_index,
                    relabel_nodes=True, 
                    num_nodes=self.data.num_nodes
                    )
            subX.append(sub_nodes)
        #self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            nodes = subX[i][:self.maxsize]
            nodes=np.array(nodes)
            #nodesub=torch.tensor(nodes)
            sub_sub.append(nodes)
            #print(nodes,'11111111111111111')
            x = self.adjust_x(nodes)

            edge = self.adjust_edge(nodes)
            self.subgraph.append(Data(x, edge))
        batch = Batch().from_data_list(self.subgraph)

        return batch,sub_sub
    
    
    
    
    
   


