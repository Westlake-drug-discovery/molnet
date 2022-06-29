import torch.nn as nn

class Activation(nn.Module):
    """
    Activation
    """
    def __init__(self, act_type, **params):
        super(Activation, self).__init__()
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leaky_relu':
            self.act = nn.LeakyReLU(**params)
        else:
            raise ValueError(act_type)
     
    def forward(self, x):
        """tbd"""
        return self.act(x)

class MLP(nn.Module):
    """
    MLP
    """
    def __init__(self, layer_num, in_size, hidden_size, out_size, act, dropout_rate):
        super(MLP, self).__init__()

        layers = []
        for layer_id in range(layer_num):
            if layer_id == 0:
                layers.append(nn.Linear(in_size, hidden_size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(Activation(act))
            elif layer_id < layer_num - 1:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(Activation(act))
            else:
                layers.append(nn.Linear(hidden_size, out_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, dim).
        """
        return self.mlp(x)

class DownstreamModel(nn.Module):
    """
    Docstring for DownstreamModel,it is an supervised 
    GNN model which predicts the tasks shown in num_tasks and so on.
    """
    def __init__(self, encoder, dataset, layer_num, hidden_size, act, dropout_rate, hid_dim=32):
        super(DownstreamModel, self).__init__()
        if dataset=="tox21":
            num_tasks = 12
            task_type = 'class'
        elif dataset == "bace":
            num_tasks = 1
            task_type = 'class'
        elif dataset == "bbbp":
            num_tasks = 1
            task_type = 'class'
        elif dataset == "toxcast":
            num_tasks = 617
            task_type = 'class'
        elif dataset == "sider":
            num_tasks = 27
            task_type = 'class'
        elif dataset == "clintox":
            num_tasks = 2
            task_type = 'class'
        elif dataset == "freesolv":
            num_tasks = 1
            task_type = 'regress'
        elif dataset == "esol":
            num_tasks = 1
            task_type = 'regress'
        elif dataset == "lipophilicity":
            num_tasks = 1
            task_type = 'regress'
        elif dataset == "hiv":
            num_tasks = 1
        elif dataset == 'muv':
            num_tasks = 17
        elif dataset == 'pcba':
            num_tasks = 128
        elif dataset == 'qm7':
            num_tasks = 1
        elif dataset == 'qm8':
            num_tasks = 12
        elif dataset == 'qm9':
            num_tasks = 12

        self.encoder = encoder
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.norm = nn.LayerNorm(hid_dim)
        self.mlp = MLP(
                layer_num,
                in_size=hid_dim,
                hidden_size=hidden_size,
                out_size=self.num_tasks,
                act=act,
                dropout_rate=dropout_rate)
        if self.task_type == 'class':
            self.out_act = nn.Sigmoid()

    def forward(self, graph):
        graph_repr = self.encoder(graph)
        graph_repr = self.norm(graph_repr)
        pred = self.mlp(graph_repr)
        if self.task_type == 'class':
            pred = self.out_act(pred)
        return pred