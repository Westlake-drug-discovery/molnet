import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_sum
from API.basic.mol_kit import CompoundKit

class GraphNorm(nn.Module):
    def __init__(self):
        super(GraphNorm, self).__init__()

    def forward(self, feature, batch):
        """graph norm"""
        nodes = torch.ones_like(batch)
        norm = scatter_sum(nodes, batch, dim=0)
        norm = torch.sqrt(norm)
        norm = norm[batch].reshape(-1,1)
        return feature / norm


class RBF(nn.Module):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = torch.tensor(centers).reshape(1,-1).float()
        self.gamma = gamma
    
    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = x.reshape(-1,1)
        return torch.exp(-self.gamma * torch.square(x - self.centers.to(x.device)))


class AtomEmbedding(nn.Module):
    """
    Atom Encoder
    """
    def __init__(self, atom_names, embed_dim):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names
        
        self.embed_list = nn.ModuleList()
        for name in self.atom_names:
            if name in CompoundKit.atom_vocab_dict.keys():
                embed = nn.Embedding(
                        CompoundKit.get_atom_feature_size(name) + 5,
                        embed_dim)
                torch.nn.init.xavier_uniform_(embed.weight.data)
                self.embed_list.append(embed)
            else:
                embed = nn.Linear(1,embed_dim)
                self.embed_list.append(embed)

    def forward(self, node_features):
        """
        Args: 
            node_features(dict of tensor): node features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[name])
        return out_embed


class AtomFloatEmbedding(nn.Module):
    """
    Atom Float Encoder
    """
    def __init__(self, atom_float_names, embed_dim, rbf_params=None):
        super(AtomFloatEmbedding, self).__init__()
        self.atom_float_names = atom_float_names
        
        if rbf_params is None:
            self.rbf_params = {
                'van_der_waals_radis': (np.arange(1, 3, 0.2), 10.0),   # (centers, gamma)
                'partial_charge': (np.arange(-1, 4, 0.25), 10.0),   # (centers, gamma)
                'mass': (np.arange(0, 2, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.atom_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, feats):
        """
        Args: 
            feats(dict of tensor): node float features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_float_names):
            x = feats[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondEmbedding(nn.Module):
    """
    Bond Encoder
    """
    def __init__(self, bond_names, embed_dim):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names
        
        self.embed_list = nn.ModuleList()
        for name in self.bond_names:
            embed = nn.Embedding(
                    CompoundKit.get_bond_feature_size(name) + 5,
                    embed_dim)
            torch.nn.init.xavier_uniform_(embed.weight.data)
            self.embed_list.append(embed)

    def forward(self, edge_features):
        """
        Args: 
            edge_features(dict of tensor): edge features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[name])
        return out_embed


class BondFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (np.arange(0, 2, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args: 
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed