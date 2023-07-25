import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F


# node features
num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3   # {unspecified, tetrahedral cw, tetrahedral ccw, other}

# edge features
num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens,  {single, double, triple, aromatic}
num_bond_direction = 3  # {–, endupright, enddownright}


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        # nn.Embedding, https://wikidocs.net/64779
        # 각각의 단어들을 정수로 맵핑한 후, vector로 맵핑한다. embedding lookup table 의 행의 수는 모든 단어의 수
        # nn.Embedding(num_embeddings=, embedding_dim=, padding_idx=)
        # num_embeddings : 임베딩 할 단어들의 수, 단어 집합의 크기
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        # xavier_uniform : 나중에 필요하면 자세히 보자
        # weight.data : embedding lookup table을 initialize
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr


    
    def forward(self, x, edge_index, edge_attr):
        #  Data(x=[29, 1] ... ) x.size(0):number of nodes
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        # aggregate 할때, 자신의 node feature도 update에 포함시키기 위해서 self loop를 만들어줌.
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        # edge feature=[bond_type, bond_direction]
        # 기존의 데이터에서는 각각의 feature가 word data로 저장되어 있음.
        # edge feature 전체를 nn.embedding을 통해서 하나의 벡터로 임베딩함.
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    # propagate에서 flow 설정이 default로 source_to_target 이기 때문에, 주변 nodes인 x_j로부터 message가 전달됨.
    # 보통은 return 할때, x_j만 return을 많이 하는데, weighted graph인 경우, node feature랑 edge attribution을 같이 전달함.
    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    # aggr_out : message 전달하고, aggregation까지 거친 output.
    # 보통 그냥 aggr_out만 return하기도 하지만, 여기서는 mlp를 통과시킨 후 return.
    def update(self, aggr_out):
        return self.mlp(aggr_out)


# unsupervised pretraining-context prediction에 사용
# input : graph data, output : node representation
class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GINConv에서 edge feature embedding후 convolution 했고, 여기서는 node feature embedding해서 넣어주기.
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        #
        # nn.ModuleList()
        # nn.Sequential이랑 다른점은, module간의 connection이 없고, forward 함수도 없다.
        # module wrapping
        self.gnns = torch.nn.ModuleList()
        
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                pass
            elif gnn_type == "gat":
                pass
            elif gnn_type == "graphsage":
                pass

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        # ? 왜 데이터 형식을 아래처럼 두개로 나누는거지?
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # shape of x : [num_of_nodes, num_of_features]
        # node features : [num_atom_type, num_chirality_tag] => 2개
        # self.x_embedding1(x[:,0]) : num_atom_type 임베딩
        # self.x_embedding2(x[:,1]) : num_chirality_tag 임베딩
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        # 위에서 embedding1,2를 같은 임베딩 space로 만들고 더함.
        # h_list는 node의 feature vector를 나열한 것.
        # ? 이거 x가 이미 [[vector1], [vector2], ... , [vector26]]이런 형태인데, [x]로 만드는거지?
        # 아, 밑에서 h_list.append(h)로 하는거 보니까, update되는 node feature embedding들을 다 저장하는건가 봄.
        # => 마지막에 최종 node_representation 뽑아낼때, 전체 결과를 concat 해서 뽑아냄.
        # 본 실험에서는 "last"로 함.(마지막 node feature == node_representation)
        h_list = [x]
        
        
        for layer in range(self.num_layer):
            # h_list[0] : 첫번째 node feature, initial node feature
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


# supervised pretrain, fine_tuning에 사용
# input : graph data, output : graph prediction
class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN module의 output으로 node_representation이 나옴.
        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        # graph pooling : node_representation, batch 을 기반으로 graph의 class를 뽑아내는 것.
        
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    # pretrained된 model 파라미터 불러오기
    # 처음 학습 할때는, 위의 self.gnn으로 학습하고,
    # fine_tuning 할때는 이 함수로 함.
    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))


    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


if __name__ == "__main__":
    pass