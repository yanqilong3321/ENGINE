import torch
from tqdm import tqdm
import numpy as np
import os
import math
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_edge_index
import yaml 
from yaml import SafeLoader
import random

from data.load import load_data
from data.sampling import collect_subgraphs, ego_graphs_sampler
from utils.peft import create_peft_config
from utils.args import Arguments
from models.encoder import GCN_Encoder, SAGE_Encoder, GIN_Encoder, MLP_Encoder, GAT_Encoder, PMLP_Encoder, GCNII_Encoder, MOE, SparseMOE
from models.encoder import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保 PyTorch 的某些操作是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    # 每个工作线程使用不同的种子，但基于全局种子
    seed = torch.initial_seed()
    np.random.seed(seed % 2**32)
    random.seed(seed)

def get_hidden_states(config):
    path = f'./llm_cache/{config.dataset}/layers'
    if not os.path.exists(os.path.join(path, 'layer_attr.pt')):
        raise FileNotFoundError(f'No cache found! Please use `python cache.py --dataset {config.dataset}` to generate it.')

    else:
        layers_hid = torch.load(os.path.join(path, 'layer_attr.pt'))

    xs = layers_hid
    return xs

def get_dataloader(data, config):
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    kwargs = {'batch_size': 256, 'num_workers': 12, 'persistent_workers': True}
    if config.sampler =='rw':
        train_graphs = collect_subgraphs(train_idx, data, walk_steps=config.walk_steps, restart_ratio=config.restart)
        val_graphs = collect_subgraphs(val_idx, data, walk_steps=config.walk_steps, restart_ratio=config.restart)
        test_graphs = collect_subgraphs(test_idx, data, walk_steps=config.walk_steps, restart_ratio=config.restart)
        train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
        val_loader = DataLoader(val_graphs, **kwargs)
        test_loader = DataLoader(test_graphs, **kwargs)
    else:
        if config.dataset in ['ogbn-arxiv', 'arxiv_2023', 'photo'] and os.path.exists(f'../subgraphs/{config.dataset}/khop-1/train.pt') and os.path.exists(f'../subgraphs/{config.dataset}/khop-1/val.pt') and os.path.exists(f'../subgraphs/{config.dataset}/khop-1/test.pt'):
            print('using cache of subgraphs')
            train_graphs = torch.load(f'../subgraphs/{config.dataset}/khop-1/train.pt')
            val_graphs = torch.load(f'../subgraphs/{config.dataset}/khop-1/val.pt')
            test_graphs = torch.load(f'../subgraphs/{config.dataset}/khop-1/test.pt')
        else:
            train_graphs = ego_graphs_sampler(train_idx, data, hop=1, sparse=(config.dataset=='ogbn-arxiv'))
            val_graphs = ego_graphs_sampler(val_idx, data, hop=1, sparse=(config.dataset=='ogbn-arxiv'))
            test_graphs = ego_graphs_sampler(test_idx, data, hop=1, sparse=(config.dataset=='ogbn-arxiv'))
            if config.dataset in ['ogbn-arxiv', 'arxiv_2023', 'photo']:
                os.makedirs(f'../subgraphs/{config.dataset}/khop-1',exist_ok = True)
                torch.save(train_graphs, f'../subgraphs/{config.dataset}/khop-1/train.pt')
                torch.save(val_graphs, f'../subgraphs/{config.dataset}/khop-1/val.pt')
                torch.save(test_graphs, f'../subgraphs/{config.dataset}/khop-1/test.pt')
            
        train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
        val_loader = DataLoader(val_graphs, **kwargs)
        test_loader = DataLoader(test_graphs, **kwargs)
    return train_loader, val_loader, test_loader

from collections import Counter
def expert_selection_distribution(selected_experts):
    all_selections = [expert for sample in selected_experts for expert in sample]
    return Counter(all_selections)

def efficient_train_eval(train_loader, val_loader, test_loader, xs, model_list, prog_list, alpha_list, exit_list,  optimizer):
    patience = config.patience
    best_acc = 0
    best_test_from_val = 0
    cnt = 0
    
    criterion =  torch.nn.CrossEntropyLoss()
    # 定义正则化超参数
    lambda_balance = getattr(config, 'lambda_balance', 1.0)  # 如果config中没有lambda_balance，则默认1.0
    lambda_entropy = getattr(config, 'lambda_entropy', 0.1)  # 如果config中没有lambda_entropy，则默认0.1
    
    for epoch in tqdm(range(config.epochs)):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            last = None
            total_loss = 0
            gates_list = []  # 存储所有专家的门控输出
            
            for i, m  in enumerate(model_list):
                m.train()
                prog_list[i].train()
                exit_list[i].train()
                alpha_list[i].requires_grad=True
                # 提取当前层的嵌入
                x = prog_list[i]((xs[i][data.original_idx.cpu()]).to(device))
                # 非第 0 层进行加权融合
                if i != 0:
                    a = torch.sigmoid(alpha_list[i] / config.T)
                    x = x * a + last * (1 - a)

                # 根据 config.encoder 判断是否需要返回 gates
                if config.encoder in {'MOE', 'SparseMOE'}:
                    out, gates = m(x, data.edge_index, return_gates=True)
                else:
                    out = m(x, data.edge_index)
        
                last = out
                config.encoder in {'MOE','SparseMOE'} and gates_list.append(gates)  # 收集门控输出

                hid_out = torch.cat([last[data.root_n_index], global_mean_pool(last, data.batch)], dim=1)
                hid_logits = exit_list[i](hid_out)
                
                total_loss += criterion(hid_logits, data.y)
            
            # 合并所有门控输出
            if config.encoder in {'MOE', 'SparseMOE'}:
                all_gates = torch.stack(gates_list, dim=0).mean(dim=0)  # 平均所有层的门控输出，形状为 [batch_size, num_experts]
                
                # 计算负载均衡和熵正则化损失
                #lb_loss = load_balance_loss(all_gates, num_experts=model_list[0].num_experts)
                #ent_loss = entropy_loss(all_gates)
            
            # 计算总损失
            #loss = total_loss + lambda_balance * lb_loss - lambda_entropy * ent_loss
            total_loss.backward()
            
            optimizer.step()

        # 打印或记录专家选择分布
        #selected_experts = [gates.argmax(dim=-1).cpu().numpy() for gates in gates_list]
        #expert_distribution = expert_selection_distribution(selected_experts)
        #print(f"Epoch {epoch+1} - Expert Selection Distribution: {expert_distribution}")

        # 评估验证和测试集
        val_acc = efficient_eval(val_loader, xs, model_list, prog_list, alpha_list, exit_list)
        test_acc = efficient_eval(test_loader, xs, model_list, prog_list, alpha_list, exit_list)
        
        if val_acc > best_acc:
            best_acc = val_acc
            cnt = 0
            best_test_from_val = test_acc
        else:
            cnt += 1
        if cnt >= patience:
            print(f'early stop at epoch {epoch}')
            return best_test_from_val
    return best_test_from_val


def efficient_eval(test_loader, xs, model_list, prog_list, alpha_list, exit_list):
    correct = 0
    total_cnt = 0
    for data in test_loader:
        total_cnt += data.batch.max().item()+1
        data = data.to(device)
            
        results = []
        last = 0
        # num_classes = data.y.max().item() + 1
        last_prediction = []
        for i, m in enumerate(model_list):
            m.eval()
            prog_list[i].eval()
            exit_list[i].eval()
            if i == 0:
                # out = m(prog_list[i](xs[i][data.original_idx]), data.edge_index)
                out = m(prog_list[i]((xs[i][data.original_idx.cpu()]).to(device)), data.edge_index)
                not_visited = torch.ones(data.batch.max()+1, device=out.device).bool()
                results = torch.rand(data.batch.max()+1, num_classes, device=out.device) # initialize results
                last_prediction = torch.ones(data.batch.max()+1, device=out.device) * -1
            else:
                a = torch.nn.functional.sigmoid(alpha_list[i]/T)
                # x = prog_list[i](xs[i][data.original_idx])*a + last*(1-a)
                x = prog_list[i]((xs[i][data.original_idx.cpu()]).to(device))*a + last*(1-a)
                out = m(x, data.edge_index)
            #distribution = expert_selection_distribution(m.selected_experts)
            #print(f"层数 {i}",distribution)
            last = out
            # dynamic early exit based on entropy
            hid_out = torch.cat([last[data.root_n_index], global_mean_pool(last, data.batch)], dim=1)
            hid_logits = exit_list[i](hid_out)
            hid_prob = torch.nn.functional.softmax(hid_logits, dim=1)
            current_prediction = hid_prob.argmax(dim=1)

            early_mask = ((current_prediction == last_prediction) == not_visited)
            results[early_mask] = hid_prob[early_mask]
            not_visited[early_mask] = False
            last_prediction = current_prediction
        results[not_visited] = hid_prob[not_visited] # samples without early exiting

        pred = results.argmax(dim=1)
        correct += (pred == data.y).sum()
    acc = int(correct) / total_cnt
    # print(f'Accuracy: {acc:.4f}') 
    return acc

def train_eval(train_loader, val_loader, test_loader, xs, model_list, prog_list,  alpha_list, exit_list, optimizer):
    patience = config.patience 
    best_acc = 0
    best_test_from_val = 0
    best_state_list = []
    cnt = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    for epoch in tqdm(range(config.epochs)):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            last = None
            for i, m in enumerate(model_list):
                m.train()
                prog_list[i].train()
                if i == 0:
                    out = m(prog_list[i]((xs[i][data.original_idx.cpu()]).to(device)), data.edge_index)
                else:
                    a = torch.nn.functional.sigmoid(alpha_list[i]/T)
                    x = prog_list[i]((xs[i][data.original_idx.cpu()]).to(device))*a + last*(1-a)
                    out = m(x, data.edge_index)
                last = out
            if hasattr(data, 'root_n_id'):
                data.root_n_index = data.root_n_id
            out = torch.cat([last[data.root_n_index], global_mean_pool(last, data.batch)], dim=1)
            out = classifier(out)
            loss = criterion(out, data.y)
            
            loss.backward()
            optimizer.step()
        val_acc = eval(val_loader, xs, model_list, prog_list,  alpha_list)
        test_acc = eval(test_loader, xs, model_list, prog_list,  alpha_list)
        
        if val_acc > best_acc:
            best_acc = val_acc
            cnt = 0
            best_test_from_val = test_acc
        else:
            cnt += 1
        if cnt >= patience:
            print(f'early stop at epoch {epoch}')
            return best_test_from_val
    # best_test_from_val = eval(test_loader, xs, model_list, prog_list,  alpha_list, zero_list)
    return best_test_from_val
    

def eval(test_loader, xs, model_list, prog_list,  alpha_list):
    correct = 0
    total = 0
    for data in test_loader:
        data = data.to(device)
        total += data.batch.max().item()+1
            
        last = None
        for i, m in enumerate(model_list):
            m.eval()
            prog_list[i].eval()
            if i == 0:
                out = m(prog_list[i]((xs[i][data.original_idx.cpu()]).to(device)), data.edge_index)
            else:
                a = torch.nn.functional.sigmoid(alpha_list[i]/T)
                x = prog_list[i]((xs[i][data.original_idx.cpu()]).to(device))*a + last*(1-a)
                # x = prog_list[i](xs[i][data.original_idx]) + last
                out = m(x, data.edge_index)
            last = out
        if hasattr(data, 'root_n_id'):
            data.root_n_index = data.root_n_id
        out = torch.cat([out[data.root_n_index], global_mean_pool(out, data.batch)], dim=1)
        out = classifier(out)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum()
    acc = int(correct) / total
    print(f'Accuracy: {acc:.4f}') 
    return acc


if __name__ == '__main__':
    config = Arguments().parse_args()
    args = yaml.load(open(config.config), Loader=SafeLoader)
    # combine args and config
    for k, v in args.items():
        config.__setattr__(k, v)
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xs = get_hidden_states(config)
    xs = [x for x in xs]
    acc_list = []
    
    for i, seed in enumerate(config.seeds):        # load data
        print(f'-------------------------seed {seed}-------------------------------')
        set_seed(seed)
        data, text, num_classes = load_data(config.dataset, use_text=True, seed=seed)
        if config.dataset == 'ogbn-products':
            edge_index, _ = to_edge_index(data.edge_index)
            data.edge_index = edge_index

        train_loader, val_loader, test_loader = get_dataloader(data, config)
        
        r=config.r # used for dimensionality reduction
        input_dim=config.input_dim # 4096
        k = int(input_dim/r)
        hidden = config.hidden_size
        layer_select = config.layer_select
        encoders = {
            'GCN_Encoder': GCN_Encoder, 
            'GAT_Encoder': GAT_Encoder, 
            'SAGE_Encoder': SAGE_Encoder, 
            'MLP_Encoder': MLP_Encoder,
            'MOE' : MOE ,
            'SparseMOE' : SparseMOE 
        }
        config.encoder='SparseMOE'
        config.num_experts = 8
        config.top_k = 1
        model_list = [encoders[config.encoder](k, config.layer_num, hidden, k, activation=config.activation, norm=config.norm, last_activation=(l !=len(layer_select)-1), dropout=config.dropout,
        **({'top_k': config.top_k, 'num_experts': config.num_experts}  # 动态添加参数
           if config.encoder in {'MOE', 'SparseMOE'} 
           else {}) ).to(device) for l in layer_select]
        prog_list = [torch.nn.Sequential(torch.nn.Linear(input_dim, k), torch.nn.LayerNorm(k), torch.nn.ReLU(), torch.nn.Linear(k,k)).to(device) for l in layer_select]
        alpha_list = [torch.nn.Parameter(torch.tensor(0.0), requires_grad=True) for l in layer_select]
        exit_list = [torch.nn.Linear(k*2, num_classes).to(device) for l in layer_select]
        classifier = torch.nn.Linear(k*2, num_classes).to(device)
        T=config.T
        lr = config.lr
        weight_decay = config.weight_decay
        
        params = []
        xs_list = []
        for i, l in enumerate(layer_select):
            params.append({'params': model_list[i].parameters(), 'lr': lr, 'weight_decay': weight_decay}) 
            params.append({'params': prog_list[i].parameters(), 'lr': lr, 'weight_decay': weight_decay}) 
            params.append({'params': alpha_list[i], 'lr': lr, 'weight_decay': weight_decay})
            params.append({'params': exit_list[i].parameters(), 'lr': lr, 'weight_decay': weight_decay})
            xs_list.append(xs[l])
        params.append({'params': classifier.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        
        optimizer = torch.optim.AdamW(params)

        # ENGINE w/ caching
        if config.early: # Early
            acc = efficient_train_eval(train_loader, val_loader, test_loader, xs_list, model_list, prog_list, alpha_list, exit_list, optimizer)
        else: 
            acc = train_eval(train_loader, val_loader, test_loader, xs_list, model_list, prog_list, alpha_list, exit_list, optimizer)
        print(acc)
        acc_list.append(acc)
        
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc*100:.2f}±{final_acc_std*100:.2f}")
    