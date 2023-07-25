import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.nn import GINConv


# ZINC(10000)
# Data(x=[29, 1], edge_index=[2, 64], edge_attr=[64], y=[1])
dataset = ZINC(root='/tmp/ZINC', subset=True)


# number of GNN layer = 5
# 
class GIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr        

def main():
    data_list = {'name': 'pey', 'phone': '010-9999-1234'}
    data_list.keys()
    print([data_list.keys()])
    # keys = [set(data.keys) for data in data_list]
    # print(keys)
    


if __name__ == '__main__':
    main()