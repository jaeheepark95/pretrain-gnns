import torch
from torch_geometric.data import Data


# https://pytorch-geometric.readthedocs.io/en/1.4.2/_modules/torch_geometric/data/batch.html

# Context prediction에서 사용함.
# 기존의 batch는 batch_size 만큼의 data graph를 묶어서 하나의 커다란 disconnected graph로 처리함.
# DataDataBatch(x=[200, 16], edge_index=[2, 1000], edge_attr=[2], y=[200], batch=[200], ptr=[3])

# 여기서는, 
# Context prediction에서 데이터에 context structure를 포함시키기 때문에, batch 할 때, 
class BatchSubstructContext(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch
        
    
    # classmethod from_data_list(data_list: List[BaseData], follow_batch: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None)
    
    
    # @staticmethod : 정적 메서드 선언. 파라미터에 self가 없음.
    # 인스턴스를 생성하지 않고, class.<함수이름>으로 바로 호출할 수 있음.
    # 외부 상태에 영향을 받지 않는 정적 함수를 만들 때 사용.
    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        
        # 
        # keys = [set(data.keys) for data in data_list]
        # keys = list(set.union(*keys))
        
        ## assert 뒤의 조건이 True가 아니면 assert error를 발생시킴.
        # assert 'batch' not in keys
        
        
        batch = BatchSubstructContext()
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]

        for key in keys:
            #print(key)
            # batch["center_substruct_idx"] = [] ...
            batch[key] = []

        # ? batch.batch = []
        # used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        # cumulative sum : 누적합
        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        
        for data in data_list:
            # If there is no context, just skip!!
            # hasattr(object, name) => bool
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                #batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ###batching for the main graph
                #for key in data.keys:
                #    if not "context" in key and not "substruct" in key:
                #        item = data[key]
                #        item = item + cumsum_main if batch.cumsum(key, item) else item
                #        batch[key].append(item)
                
                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)
                

                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct   
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        #batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        # .contiguous() : tensor의 shape를 조작하면서, 메모리 주소 순서가 뒤죽박죽이 됨.
        # contiguous()로 메모리 순서를 다시 맞춰 줄 수 있음.
        return batch.contiguous()

    # 
    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    #
    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]
    
    # @property : 어떤 인스턴스 변수에 대한 getter method
    # 인스턴스.num_graph 사용시 아래 함수가 호출되어, number of graph를 반환함.
    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
