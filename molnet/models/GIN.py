from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_sum, scatter_mean
from .common import GraphNorm, AtomEmbedding, BondEmbedding, BondFloatRBF


class GIN(MessagePassing):
    def __init__(self, emb_dim, eps=0.1):
        super(GIN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_attr=None):
        src, dst = edge_index[0], edge_index[1]
        if edge_attr is None:
            edge_attr = x[src]
        else:
            edge_attr = edge_attr + x[src]
        node_feat = scatter_sum(edge_attr, dst, dim=0, dim_size=x.shape[0])
        node_feat = node_feat+(1+self.eps)*x
        node_feat = self.mlp(node_feat)
        return node_feat


class GINBlock(nn.Module):
    def __init__(self, embed_dim, dropout_rate, last_act):
        super().__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GIN(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_norm = GraphNorm()
        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, node_hidden, edge_index, edge_hidden, batch):
        out = self.gnn(node_hidden, edge_index, edge_hidden)
        out = self.norm(out)
        out = self.graph_norm(out, batch)
        if self.last_act:
            out = self.act(out)
        out = self.dropout(out)
        out = out + node_hidden
        return out


class GINNet(nn.Module):
    def __init__(self, atom_names, bond_names, bond_float_names, layer_num, embed_dim, dropout_rate, pool='mean'):
        super().__init__()
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.layer_num = layer_num
        self.pool = pool
        self.embed_dim = embed_dim
        self.init_atom_embedding = AtomEmbedding(atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(bond_float_names, self.embed_dim)

        self.gnn_blocks = nn.ModuleList()
        for layer_id in range(self.layer_num-1):
            self.gnn_blocks.append(GINBlock(embed_dim, dropout_rate,True))
        self.gnn_blocks.append(GINBlock(embed_dim, dropout_rate,False))

    def forward(self, graph):
        node_feat = {name: graph.get(name) for name in self.atom_names }
        node_hidden = self.init_atom_embedding(node_feat)
        edge_feat = {name: graph.get(name) for name in self.bond_names+ self.bond_float_names}
        bond_embed = self.init_bond_embedding(edge_feat)
        edge_hidden = bond_embed + self.init_bond_float_rbf(edge_feat)

        for layer_id in range(self.layer_num):
             node_hidden = self.gnn_blocks[layer_id](node_hidden, graph.edge_index, edge_hidden, graph.batch)
        
        if self.pool == 'mean':
            graph_hidden = scatter_mean(node_hidden, graph.batch, dim=0)
        if self.pool == 'sum':
            graph_hidden = scatter_sum(node_hidden, graph.batch, dim=0)
        return graph_hidden