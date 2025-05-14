import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool, GraphNorm, GCN2Conv
from torch.nn import BatchNorm1d, Identity, LayerNorm
import torch.nn as nn
import numpy as np
import random
from utils.args import Arguments
from utils.register import register
from torch_geometric.utils import get_laplacian
# from .conv import *

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

set_random_seed(7)
args = Arguments().parse_args()


def get_activation(name: str):
        activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }
        return activations[name]
    

def get_norm(name: str):
    norms = {
        'id': Identity,
        'bn': BatchNorm1d,
        'ln': LayerNorm
    }
    return norms[name]
    
    
@register.encoder_register
class GCN_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(GCN_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GCNConv(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(GCNConv(hidden_size, hidden_size))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(GCNConv(hidden_size, output_dim))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))

        else: # one layer gcn
            self.convs.append(GCNConv(input_dim, output_dim)) 
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.norms[i](self.convs[i](x, edge_index, edge_weight))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
            

@register.encoder_register
class GCNII_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(GCNII_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GCN2Conv(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(GCN2Conv(hidden_size, hidden_size))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(GCN2Conv(hidden_size, output_dim))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))

        else: # one layer gcn
            self.convs.append(GCN2Conv(input_dim, output_dim)) 
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.norms[i](self.convs[i](x, edge_index, edge_weight))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
                

@register.encoder_register
class SAGE_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(SAGE_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(SAGEConv(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(SAGEConv(hidden_size, hidden_size))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(SAGEConv(hidden_size, output_dim))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))
        else: # one layer gcn
            self.convs.append(SAGEConv(input_dim, output_dim)) 
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.norms[i](self.convs[i](x, edge_index, edge_weight))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
                

@register.encoder_register               
class GIN_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(GIN_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        
        self.readout = global_mean_pool
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, hidden_size)))) 
            for i in range(layer_num-2):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, hidden_size))))
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, output_dim))))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))

        else: # one layer gcn
            self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, hidden_size)))) 
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, **kwargs):
        for i in range(self.layer_num):
            x = self.norms[i](self.convs[i](x, edge_index))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        # out_readout = self.readout(x, batch, batch_size)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
                

@register.encoder_register
class GAT_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(GAT_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm
        self.heads = 8

        self.convs = ModuleList()
        self.norms = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GATConv(input_dim, hidden_size, heads=self.heads, dropout=dropout)) 
            for i in range(layer_num-2):
                self.convs.append(GATConv(hidden_size*self.heads, hidden_size, heads=self.heads, dropout=dropout))
            self.convs.append(GATConv(hidden_size*self.heads, output_dim, heads=1, dropout=dropout))
            
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size*self.heads))
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
        else: # one layer gcn
            self.heads=1
            self.convs.append(GATConv(input_dim, output_dim, heads=self.heads, dropout=dropout)) 
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, **kwargs):
        for i in range(self.layer_num):
            x = self.norms[i](self.convs[i](x, edge_index))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
            

@register.encoder_register
class MLP_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(MLP_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        
        self.readout = global_mean_pool
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(nn.Linear(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(nn.Linear(hidden_size, hidden_size))
            self.convs.append(nn.Linear(hidden_size, output_dim))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))

        else: # one layer gcn
            self.convs.append(nn.Linear(input_dim, output_dim))
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index=None, **kwargs):
        for i in range(self.layer_num):
            x = self.norms[i](self.convs[i](x))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        # out_readout = self.readout(x, batch, batch_size)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()
            
            
@register.encoder_register
class PMLP_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(PMLP_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        
        self.readout = global_mean_pool
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(nn.Linear(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(nn.Linear(hidden_size, hidden_size))
            self.convs.append(nn.Linear(hidden_size, output_dim))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))

        else: # one layer gcn
            self.convs.append(nn.Linear(input_dim, hidden_size))
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, **kwargs):
        for i in range(self.layer_num):
            x = self.convs[i](x)
            if not self.training:
                x = gcn_conv(x, edge_index) 
            x = self.norms[i](x)
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()


# 定义门控网络
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)  # softmax 输出权重，表示每个专家的选择概率

# 定义MOE模型
class MOE(nn.Module):
    def __init__(self,  input_dim=128, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True, rho=0.5,num_experts=4, top_k=2):
        super(MOE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k  # 选择Top-k个专家
        self.rho = rho  # 训练频率正则化超参数
        self.experts = nn.ModuleList([
            SAGE_Encoder(
                input_dim=input_dim, 
                layer_num=layer_num, 
                hidden_size=hidden_size,
                output_dim=output_dim, 
                activation=activation, 
                dropout=dropout,
                norm=norm, 
                last_activation=last_activation
            ) for _ in range(num_experts)
        ])

        self.gating_network = GatingNetwork(input_dim, num_experts)
        # 初始化每个专家的训练计数（可选）
        # self.training_counts = torch.zeros(num_experts, dtype=torch.float)

    def forward(self, x, edge_index, return_gates=False):
        """
        前向传播方法。
        :param x: 输入特征
        :param edge_index: 图的边索引
        :param return_gates: 是否返回门控权重
        :return: 输出结果，如果 return_gates=True，则返回 (output, gate_weights)
        """
        # 获取门控网络的输出，即每个专家的选择权重
        gate_weights = self.gating_network(x)  # [batch_size, num_experts]

        
        # 对专家选择权重进行排序，选择Top-k个专家
        top_k_values, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # [batch_size, top_k]

        # 记录选择的专家
        self.selected_experts = top_k_indices.detach().cpu().tolist()  # 保存为列表，方便后续分析

        # 获取所有专家的输出
        expert_outputs = torch.stack([expert(x, edge_index) for expert in self.experts], dim=1)  # [batch_size, num_experts, output_dim]
        
        # 获取Top-k专家的输出
        # 需要扩展top_k_indices以匹配expert_outputs的最后一维
        expanded_top_k_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))  # [batch_size, top_k, output_dim]
        top_k_expert_outputs = torch.gather(expert_outputs, 1, expanded_top_k_indices)  # [batch_size, top_k, output_dim]
        
        # 加权求和Top-k专家的输出
        weighted_top_k = top_k_expert_outputs * top_k_values.unsqueeze(-1)  # [batch_size, top_k, output_dim]
        output = torch.sum(weighted_top_k, dim=1)  # [batch_size, output_dim]
        
        if return_gates:
            return output, gate_weights
        return output
from torch_geometric.utils import dense_to_sparse

class SparseGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, score_func="sigmoid", route_scale=1.0,
                 kernel_type='rbf', kernel_mul=2.0, kernel_num=5, use_gnn=True, gnn_hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_scale = route_scale
        self.use_gnn = use_gnn

        # 专家向量
        self.expert_vector = nn.Parameter(torch.empty(num_experts, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_experts))

        nn.init.kaiming_uniform_(self.expert_vector, nonlinearity='linear')

        # 结构信息 GNN
        if self.use_gnn:
            self.structure_encoder = GCNConv(input_dim, gnn_hidden_dim)
            self.structure_fc = nn.Linear(gnn_hidden_dim, input_dim)

        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制语义 vs 结构信息的权重
    def forward(self, x, edge_index, ppr_values=None):
        batch_size = 256

        # 计算语义相似度
        semantic_scores = x @ self.expert_vector.t() + self.bias  # [batch_size, num_experts]

        # 计算结构相似度（如果启用 GNN）
        if self.use_gnn:
            structure_features = self.structure_encoder(x, edge_index)  # GCN 计算结构信息
            structure_features = self.structure_fc(structure_features)
            structure_scores = structure_features @ self.expert_vector.t()  # [batch_size, num_experts]

        if ppr_values !=None:

            print(ppr_values.shape)
            print(edge_index.shape)
            print(x.shape)
            print(semantic_scores.shape)
            #edge_index, edge_weight = dense_to_sparse(ppr_values)  # 转换为稀疏格式
            #print(edge_index.shape)
            #exit()
            structure_scores = x @ self.expert_vector.t()
        else:
            structure_scores = torch.zeros_like(semantic_scores)
            
        # 结合语义和结构信息
        #final_scores = self.alpha * semantic_scores + (1 - self.alpha) * structure_scores
        final_scores = self.alpha * semantic_scores + (1 - self.alpha) * structure_scores

        # 归一化
        if self.score_func == "softmax":
            original_scores = torch.softmax(final_scores, dim=-1)
        elif self.score_func == "sigmoid":
            original_scores = torch.sigmoid(final_scores)
        else:
            raise ValueError(f"Unsupported score function: {self.score_func}")

        # 选择 Top-K 专家
        top_k_weights, top_k_indices = torch.topk(original_scores, self.top_k, dim=-1)

        # Sigmoid 需要手动归一化
        if self.score_func == "sigmoid":
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 应用路由缩放因子
        top_k_weights = top_k_weights * self.route_scale

        # 生成稀疏权重矩阵
        sparse_weights = torch.zeros_like(original_scores)
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)

        return sparse_weights, top_k_indices


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SparseMOE(nn.Module):
    def __init__(self, input_dim=128, layer_num=2, hidden_size=128, output_dim=128, 
                 activation="relu", dropout=0.5, norm='id', last_activation=True, 
                 rho=0.5, num_experts=4, top_k=2, score_func="softmax", route_scale=1.0,
                 use_gnn=False, gnn_hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Share 专家（始终激活）
        self.share_expert = SAGE_Encoder(
            input_dim=input_dim, 
            layer_num=layer_num, 
            hidden_size=hidden_size,
            output_dim=output_dim, 
            activation=activation, 
            dropout=dropout,
            norm=norm, 
            last_activation=last_activation
        )

        # Router 专家（由门控选择）
        self.router_experts = nn.ModuleList([
            SAGE_Encoder(
                input_dim=input_dim, 
                layer_num=layer_num, 
                hidden_size=hidden_size,
                output_dim=output_dim, 
                activation=activation, 
                dropout=dropout,
                norm=norm, 
                last_activation=last_activation
            ) for _ in range(num_experts)
        ])

        # 门控网络（支持 Softmax/Sigmoid 和路由缩放）
        self.gating_network = SparseGatingNetwork(
            input_dim, num_experts, top_k, score_func, route_scale,
            use_gnn=use_gnn, gnn_hidden_dim=gnn_hidden_dim
        )
        self.register_buffer('expert_counts', torch.zeros(num_experts))

    def forward(self, x, edge_index, return_gates=False,ppr_values=None):
        # Share 专家输出（始终激活）
        share_output = self.share_expert(x, edge_index)  # [batch_size, output_dim]
        # Router 专家输出（稀疏激活）
        gate_weights, expert_indices = self.gating_network(x, edge_index,ppr_values)  # 需要传递 edge_index
        router_outputs = torch.stack([expert(x, edge_index) for expert in self.router_experts], dim=1)  # [batch_size, num_experts, output_dim]

        # 高效加权（仅计算激活的专家）
        router_output = torch.zeros_like(share_output)
        for i in range(self.top_k):
            idx = expert_indices[:, i]
            router_output += gate_weights[:, i].unsqueeze(-1) * router_outputs[torch.arange(x.size(0)), idx]

        # 更新专家选择统计
        self.expert_counts += gate_weights.sum(dim=0).detach()

        # 最终输出 = Share 输出 + Router 输出
        moe_output = share_output + router_output


        if return_gates:
            return moe_output, gate_weights
        return moe_output

    def reset_expert_counts(self):
        self.expert_counts.zero_()

    def load_balance_loss(self, eps=1e-10):
        """负载均衡损失（基于专家选择频率 + 熵正则化）"""
        prob = self.expert_counts / (self.expert_counts.sum() + eps)
        entropy_loss = -torch.sum(prob * torch.log(prob + eps))  # 熵最大化
        return entropy_loss



class BaseModel(nn.Module):
    """共享的基础网络"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.feature_extractor(x)

class TaskIncrementalModel(nn.Module):
    """任务增量学习模型"""
    def __init__(self, initial_classes):
        super().__init__()
        self.base = BaseModel()
        # 每个任务有自己的分类头
        self.heads = nn.ModuleList([nn.Linear(128, initial_classes)])
        
    def add_task(self, num_classes):
        # 添加新任务时增加一个新头
        self.heads.append(nn.Linear(128, num_classes))
        
    def forward(self, x, task_id):
        features = self.base(x)
        return self.heads[task_id](features)

class ClassIncrementalModel(nn.Module):
    """类别增量学习模型"""
    def __init__(self, initial_classes):
        super().__init__()
        self.base = BaseModel()
        # 单一分类头，输出维度会扩展
        self.head = nn.Linear(128, initial_classes)
        self.total_classes = initial_classes
        
    def add_classes(self, num_new_classes):
        # 扩展分类头的输出维度
        old_head = self.head
        self.total_classes += num_new_classes
        self.head = nn.Linear(128, self.total_classes)
        
        # 复制旧权重（实际实现中可能需要更复杂的权重初始化）
        with torch.no_grad():
            self.head.weight[:old_head.out_features] = old_head.weight
            self.head.bias[:old_head.out_features] = old_head.bias
            
    def forward(self, x):
        features = self.base(x)
        return self.head(features)
