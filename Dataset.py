import torch
from torch_geometric.datasets import QM9
import torch_geometric


class DatasetVirtualNode:
    def __init__(self, root: str):
        self.data_list = []
        self.dataset_qm9 = QM9(root=root)
        virtual_node = torch.ones((1, 11))          # virtual node represents molecule; dummy embedding
        virtual_edge_attr = torch.zeros((1, 4))     # virtual node edges; one hot => all zeros => no bonds
        virtual_node_pos = torch.zeros((1, 3))      # virtual node position set to origin
        virtual_node_z = torch.Tensor([-1])         # virtual node z
        for molecule_i in range(len(self.dataset_qm9)):
            self.data_list.append(torch_geometric.data.Data(
                x=torch.cat((self.dataset_qm9[molecule_i].x, virtual_node), dim=0),
                edge_index=self.add_virtual_node_edges(molecule_i=molecule_i),
                edge_attr=torch.cat((self.dataset_qm9[molecule_i].edge_attr, virtual_edge_attr), dim=0),
                pos=torch.cat((self.dataset_qm9[molecule_i].pos, virtual_node_pos), dim=0),
                z=torch.cat((self.dataset_qm9[molecule_i].z, virtual_node_z), dim=-1)
            ))
            if molecule_i % 1000 == 0:
                print("Augmentation completion: "
                      "{percentage:.3f}%".format(percentage=((molecule_i / len(self.dataset_qm9)) * 100)))
        del self.dataset_qm9

    def add_virtual_node_edges(self, molecule_i: int):
        return torch.cat((
            self.dataset_qm9[molecule_i].edge_index,
            torch.cat((
                torch.arange(0, self.dataset_qm9[molecule_i].x.shape[0]).unsqueeze(0),
                torch.full((1, self.dataset_qm9[molecule_i].x.shape[0]),
                           fill_value=self.dataset_qm9[molecule_i].x.shape[0])
            ), dim=0)
        ), dim=1)

    def __getitem__(self, index):
        data = self.data_list[index]
        node_embeds = data.x
        edge_index = data.edge_index
        labels = data.y
        return node_embeds, edge_index, labels

    def __len__(self):
        return len(self.data_list)
