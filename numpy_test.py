import numpy as np
import torch


data_list = {"node_num" : "30", "edge_index" : "212"}

keys = [set(data.keys) for data in data_list]
keys = list(set.union(*keys))

print(keys)


