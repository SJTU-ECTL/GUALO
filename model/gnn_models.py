import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, SAGEConv, SAGPooling
from torch_geometric.nn import global_max_pool, global_mean_pool


NUM_NODE_FEATURES = 5


class NodeEncoder(nn.Module):


    def __init__(self, emb_dim):
        super().__init__()
        self.node_type_embedding = nn.Embedding(NUM_NODE_FEATURES, emb_dim)
        nn.init.xavier_uniform_(self.node_type_embedding.weight.data)

    def forward(self, x):
        x_embedding = self.node_type_embedding(x)
        x_embedding = x_embedding.reshape(x_embedding.shape[0], -1)
        return x_embedding


class GNN(nn.Module):


    def __init__(self, node_encoder, input_dim, emb_dim=128, n_layers=5, dropout=0.5,
                 normalization="running"):
        super().__init__()
        if normalization not in ("running", "per_graph"):
            raise ValueError("normalization must be running or per_graph")
        self.normalization = normalization
        self.node_encoder = node_encoder
        self.pool1 = SAGPooling(emb_dim, ratio=0.8)
        self.pool2 = SAGPooling(emb_dim, ratio=0.8)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            in_channels = input_dim if i == 0 else emb_dim
            self.convs.append(SAGEConv(in_channels, emb_dim))
            self.bns.append(nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        x = self.node_encoder(x)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if self.normalization == "per_graph":


                normalized = torch.empty_like(x)
                bn = self.bns[i]
                for graph_id in batch.unique():
                    mask = batch == graph_id
                    if int(mask.sum()) < 2:
                        raise ValueError("Per-graph normalization needs at least two nodes")
                    normalized[mask] = F.batch_norm(x[mask], None, None, bn.weight, bn.bias,
                                                    training=True, momentum=0.0, eps=bn.eps)
                x = normalized
            else:
                x = self.bns[i](x)
            x = F.relu(x)

        x_global = torch.cat(
            (global_mean_pool(x, batch), global_max_pool(x, batch)), dim=-1
        )
        return x_global


class GIN(nn.Module):


    def __init__(self, node_encoder, input_dim, emb_dim=128, n_layers=4, dropout=0.5):
        super().__init__()
        self.node_encoder = node_encoder
        self.pool1 = SAGPooling(emb_dim, ratio=0.8)
        self.pool2 = SAGPooling(emb_dim, ratio=0.8)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            in_channels = input_dim if i == 0 else emb_dim
            mlp = nn.Sequential(
                nn.Linear(in_channels, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            self.convs.append(GINConv(mlp, eps=0.0, train_eps=False))
            self.bns.append(nn.BatchNorm1d(emb_dim))
        self.dropout = dropout

    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        x = self.node_encoder(x)

        for i in range(len(self.convs)):
            x = F.relu(self.bns[i](self.convs[i](x, edge_index)))
            if i == 0:
                x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            elif i == len(self.convs) - 1:
                x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_global = torch.cat(
            (global_mean_pool(x, batch), global_max_pool(x, batch)), dim=-1
        )
        return x_global
