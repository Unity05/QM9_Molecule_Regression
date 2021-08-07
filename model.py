import torch
import torch.nn.functional as F
import torch_geometric.nn as nn


class DipoleNet(torch.nn.Module):
    def __init__(self):
        super(DipoleNet, self).__init__()
        # GNN
        self.conv_1 = DipoleNetLayer(11, 19)
        self.conv_2 = DipoleNetLayer(19, 19)
        self.conv_3 = DipoleNetLayer(19, 19)
        # MLP Regression
        self.fc_0 = torch.nn.Linear(19, 10)
        self.fc_1 = torch.nn.Linear(10, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = F.relu(self.conv_1(x, edge_index))
        x = F.dropout(x, training=self.training, p=0.1)
        x = F.relu(self.conv_2(x, edge_index) + x)
        x = F.relu(self.conv_3(x, edge_index) + x)
        x = F.dropout(x, training=self.training, p=0.1)
        # We only need the embedding of the virtual node.
        x = DipoleNet.get_virtual_node_position(
            node_embeddings=x,
            batch=batch
        )
        x = F.dropout(x, training=self.training, p=0.1)
        x = F.relu(self.fc_0(x))
        x = F.dropout(x, training=self.training, p=0.1)
        x = F.relu(self.fc_1(x))
        return x

    @staticmethod
    def get_virtual_node_position(node_embeddings: torch.Tensor, batch: torch.Tensor):
        relative_positions = torch.bincount(batch)
        for i in range(1, relative_positions.shape[0]):
            relative_positions[i] = relative_positions[i] + relative_positions[i - 1]
        relative_positions = relative_positions - 1
        return node_embeddings[relative_positions]


class DipoleNetLayer(nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DipoleNetLayer, self).__init__(aggr='add')
        self.fc_0 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.fc_1 = torch.nn.Linear(in_features=(in_channels + out_channels), out_features=out_channels)

    def forward(self, x, edge_index):
        x_1 = self.neighbor_aggr(x=x, edge_index=edge_index)
        x = self.final_aggr(x_0=x, x_1=x_1)
        return x

    def neighbor_aggr(self, x, edge_index):
        # Linear transformation of node features.
        x = self.fc_0(x)
        x = self.propagate(edge_index=edge_index, x=x, norm=None)   # We don't want to normalize here.
        return x

    def final_aggr(self, x_0, x_1):
        x = torch.cat((x_0, x_1), dim=-1)
        x = self.fc_1(x)
        return x
