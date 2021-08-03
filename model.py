import torch
import torch.nn.functional as F
import torch_geometric.nn as nn


class DipoleNet(torch.nn.Module):
    def __init__(self):
        super(DipoleNet, self).__init__()
        # GNN
        self.conv_1 = nn.GCNConv(11, 15)
        self.conv_2 = nn.GCNConv(15, 19)
        # MLP Regression
        self.fc_0 = torch.nn.Linear(19, 10)
        self.fc_1 = torch.nn.Linear(10, 5)
        self.fc_2 = torch.nn.Linear(5, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv_1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv_2(x, edge_index))
        # We only need the embedding of the virtual node.
        x = DipoleNet.get_virtual_node_position(
            node_embeddings=x,
            batch=batch
        )
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_2(x)
        return x

    @staticmethod
    def get_virtual_node_position(node_embeddings: torch.Tensor, batch: torch.Tensor):
        relative_positions = torch.bincount(batch)
        for i in range(1, relative_positions.shape[0]):
            relative_positions[i] = relative_positions[i] + relative_positions[i - 1]
        relative_positions = relative_positions - 1
        return node_embeddings[relative_positions]
