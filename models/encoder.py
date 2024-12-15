import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool, GraphNorm, GCN2Conv
from torch.nn import BatchNorm1d, Identity, LayerNorm
import torch.nn as nn
import numpy as np
import random

from utils.register import register
# from .conv import *

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

set_random_seed(7)

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
    def __init__(self, num_experts=4, top_k=1, input_dim=128, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True,rho=0.5):
        super(MOE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k  # 选择Top-k个专家
        self.rho = rho  # 训练频率正则化超参数
        #self.experts = nn.ModuleList([GCNExpert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.experts = nn.ModuleList([SAGE_Encoder(input_dim=input_dim, layer_num=layer_num, hidden_size=hidden_size,
                        output_dim=output_dim, activation=activation, dropout=dropout,
                        norm=norm, last_activation=last_activation) for _ in range(num_experts)])

        self.gating_network = GatingNetwork(input_dim, num_experts)
        # 初始化每个专家的训练计数
        #self.training_counts = torch.zeros(num_experts, dtype=torch.float)  # 每个专家的训练样本数

    def update_training_counts(self, selected_experts):
        """
        更新专家模型的训练样本计数。
        :param selected_experts: 当前批次中被选择的专家模型索引（列表或张量）。
        """
        for expert in selected_experts:
            self.training_counts[expert] += 1

    def recalibrate_scores(self, gate_weights):
        """
        根据训练频率正则化公式调整专家模型的得分。
        :param gate_weights: 原始的专家模型得分（形状为 [batch_size, num_experts]）。
        :return: 调整后的专家模型得分（形状与输入一致）。
        """
        # 计算每个专家的归一化训练频率
        training_frequencies = self.training_counts / self.training_counts.sum()

        # 校正因子
        recalibration_factors = 1.0 - training_frequencies.unsqueeze(0)  # 形状 [1, num_experts]

        # 确保校正因子与 gate_weights 在同一设备上
        recalibration_factors = recalibration_factors.to(gate_weights.device)

        # 使用公式调整得分
        recalibrated_scores = gate_weights * (recalibration_factors * self.rho + (1.0 - self.rho) / 2)
        return recalibrated_scores
        
    def forward(self, x, edge_index):
        # 获取门控网络的输出，即每个专家的选择权重
        gate_weights = self.gating_network(x)

        # 调整得分以避免 "赢家通吃" 情况
        #gate_weights = self.recalibrate_scores(gate_weights)
        
        # 对专家选择权重进行排序，选择Top-k个专家
        top_k_values, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)

        # 记录选择的专家
        self.selected_experts = top_k_indices.detach().cpu().tolist()  # 保存为列表，方便后续分析

        # 更新训练计数
        #self.update_training_counts(top_k_indices.flatten())


        # 获取Top-k专家的输出
        expert_outputs = [expert(x, edge_index) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
        
        # 使用选定的Top-k专家进行加权求和
        top_k_expert_outputs = expert_outputs.gather(1, top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))  # 获取Top-k专家的输出
        output = torch.sum(top_k_expert_outputs * top_k_values.unsqueeze(-1), dim=1)  # 加权求和
        
        return output