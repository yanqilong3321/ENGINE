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
from models.encoder import GCN_Encoder, SAGE_Encoder, GIN_Encoder, MLP_Encoder, GAT_Encoder, PMLP_Encoder, GCNII_Encoder,MOE,SparseMOE


import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
import copy







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

def get_dataloader(data, config,task_id):
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    train_path = f'../subgraphs/{config.dataset}/khop-1/cl_train_task{task_id}.pt'
    val_path   = f'../subgraphs/{config.dataset}/khop-1/cl_val_task{task_id}.pt'
    test_path  = f'../subgraphs/{config.dataset}/khop-1/cl_test_task{task_id}.pt'
    kwargs = {'batch_size': 256, 'num_workers': 12, 'persistent_workers': True}
    if config.sampler =='rw':
        train_graphs = collect_subgraphs(train_idx, data, walk_steps=config.walk_steps, restart_ratio=config.restart)
        val_graphs = collect_subgraphs(val_idx, data, walk_steps=config.walk_steps, restart_ratio=config.restart)
        test_graphs = collect_subgraphs(test_idx, data, walk_steps=config.walk_steps, restart_ratio=config.restart)
        train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
        val_loader = DataLoader(val_graphs, **kwargs)
        test_loader = DataLoader(test_graphs,**kwargs)
    else:
        if config.dataset in ['ogbn-arxiv', 'arxiv_2023', 'photo'] and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            print('Using cached subgraphs for Task', task_id)
            train_graphs = torch.load(train_path)
            val_graphs = torch.load(val_path)
            test_graphs = torch.load(test_path)
        else:
            train_graphs = ego_graphs_sampler(train_idx, data, hop=1, sparse=(config.dataset=='ogbn-arxiv'))
            val_graphs = ego_graphs_sampler(val_idx, data, hop=1, sparse=(config.dataset=='ogbn-arxiv'))
            test_graphs = ego_graphs_sampler(test_idx, data, hop=1, sparse=(config.dataset=='ogbn-arxiv'))
            if config.dataset in ['ogbn-arxiv', 'arxiv_2023', 'photo']:
                os.makedirs(f'../subgraphs/{config.dataset}/khop-1', exist_ok = True)
                torch.save(train_graphs, train_path)
                torch.save(val_graphs,   val_path)
                torch.save(test_graphs,  test_path)
            
        train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
        val_loader = DataLoader(val_graphs, **kwargs)
        test_loader = DataLoader(test_graphs, **kwargs)
    return train_loader, val_loader, test_loader

from collections import Counter
def expert_selection_distribution(selected_experts):
    all_selections = [expert for sample in selected_experts for expert in sample]
    return Counter(all_selections)

# 定义负载均衡损失函数
def load_balance_loss(gates, num_experts):
    avg_gate = torch.mean(gates, dim=0)  # [num_experts]
    balance_loss = torch.sum((avg_gate - 1.0 / num_experts) ** 2)
    return balance_loss

# 定义熵损失函数
def entropy_loss(gates):
    entropy = -torch.sum(gates * torch.log(gates + 1e-10), dim=1)  # [batch_size]
    return torch.mean(entropy)



def efficient_train_eval(train_loader, val_loader, test_loader, xs, model_list, prog_list, alpha_list, exit_list, optimizer, loss_fn,task_id,task_class_splits):
    patience = config.patience
    best_acc = 0
    best_test_from_val = 0
    cnt = 0
    
    
    lambda_balance = getattr(config, 'lambda_balance', 0.01)  # 如果config中没有lambda_balance，则默认1.0
    lambda_entropy = getattr(config, 'lambda_entropy', 0.01)  # 如果config中没有lambda_entropy，则默认0.1
    criterion = loss_fn
    best_model_state = []
    best_prog_state = []
    best_exit_state = []
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
                if config.encoder == 'MOE' or config.encoder == 'SparseMOE':
                    out, gates = m(x, data.edge_index, return_gates=True)
                else:
                    out = m(x, data.edge_index)

                last = out
                if config.encoder == 'MOE' or config.encoder == 'SparseMOE':
                    gates_list.append(gates)  # 收集门控输出
                hid_out = torch.cat([last[data.root_n_index], global_mean_pool(last, data.batch)], dim=1)

                hid_logits = exit_list[i](hid_out)


                task_classes = task_class_splits[task_id]
                num_classes = len(task_classes)  # 当前任务的类别数量
                class_offset = task_classes[0]  # 当前任务的类别起始值

                #relevant_logits = hid_logits[:, -num_classes:]
                #relevant_labels = data.y - class_offset  # 将标签映射到 [0, num_classes]
                #total_loss +=  criterion(relevant_logits, relevant_labels)


                #print("hid_logits shape:", hid_logits.shape)
                #print("data.y range:", torch.min(data.y).item(), "to", torch.max(data.y).item())

                total_loss +=  criterion(hid_logits, data.y)


            # 合并所有门控输出
            if config.encoder == 'MOE' or config.encoder == 'SparseMOE':
                all_gates = torch.stack(gates_list, dim=0).mean(dim=0)  # 平均所有层的门控输出，形状为 [batch_size, num_experts]
            
            # 计算负载均衡和熵正则化损失
            #lb_loss = load_balance_loss(all_gates, num_experts=model_list[0].num_experts)
            #ent_loss = entropy_loss(all_gates)
            
            # 计算总损失
            #loss = total_loss + lambda_balance * lb_loss - lambda_entropy * ent_loss
            #loss.backward(retain_graph=True)
            total_loss.backward()
            optimizer.step()

        # 打印或记录专家选择分布
        #selected_experts = [gates.argmax(dim=-1).cpu().numpy() for gates in gates_list]
        #expert_distribution = expert_selection_distribution(selected_experts)
        #print(f"Epoch {epoch+1} - Expert Selection Distribution: {expert_distribution}")


        val_acc = efficient_eval(val_loader, xs, model_list, prog_list, alpha_list, exit_list,task_id)
        test_acc = efficient_eval(test_loader, xs, model_list, prog_list, alpha_list, exit_list,task_id)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_test_from_val = test_acc
            cnt = 0
            # 保存最佳模型状态
            best_model_state = copy.deepcopy([m.state_dict() for m in model_list])
            best_prog_state = copy.deepcopy([p.state_dict() for p in prog_list])
            best_exit_state = copy.deepcopy([e.state_dict() for e in exit_list])
            # 保存最佳模型状态
            '''
            torch.save({
                'model_list': [m.state_dict() for m in model_list],
                'prog_list': [p.state_dict() for p in prog_list],
                'exit_list': [e.state_dict() for e in exit_list],
            }, "./best_model_states.pth")
            '''
        else:
            cnt += 1
        if cnt >= patience:
            print(f'early stop at epoch {epoch}')
            #print(best_test_from_val)
            for model, state in zip(model_list, best_model_state):
                model.eval()
                model.load_state_dict(state)
                #for param in model.parameters():
                    #print(param.mean())  # 或其他统计量
            for prog, state in zip(prog_list, best_prog_state):
                prog.eval()
                prog.load_state_dict(state)
            for exit_layer, state in zip(exit_list, best_exit_state):
                exit_layer.eval()
                exit_layer.load_state_dict(state)
            return best_test_from_val
    return best_test_from_val

def efficient_eval1(test_loader, xs, model_list, prog_list, alpha_list, exit_list,task_id):
    correct = 0
    total_cnt = 0
    with torch.no_grad():
        for data in test_loader:
            batch_size = data.batch.max().item() + 1
            total_cnt += batch_size
            data = data.to(device)
            
            last = None
            num_classes = len(data.y.unique())
            class_offset = data.y.unique()[0]  # 当前任务的类别起始值
            results = torch.zeros(batch_size, num_classes, device=device)
            last_prediction = torch.ones(batch_size, device=device) * -1  # 初始化为 -1
            
            for i, m in enumerate(model_list):
                m.eval()
                prog_list[i].eval()
                exit_list[i].eval()
                alpha_list[i].requires_grad=False
                # 提取当前层的嵌入
                x = prog_list[i]((xs[i][data.original_idx.cpu()]).to(device))
                # 非第 0 层进行加权融合
                if i != 0:
                    a = torch.sigmoid(alpha_list[i] / config.T)
                    x = x * a + last * (1 - a)

                # 根据 config.encoder 判断是否需要返回 gates
                if config.encoder == 'MOE':
                    out, gates = m(x, data.edge_index, return_gates=True)
                    # 此处可以选择对 gates 进行进一步处理或收集
                    # 例如: gates_list.append(gates)
                else:
                    out = m(x, data.edge_index)
                last = out
                hid_out = torch.cat([last[data.root_n_index], global_mean_pool(last, data.batch)], dim=1)
                hid_logits = exit_list[i](hid_out)  

                relevant_logits = hid_logits[:, :num_classes]
                hid_prob = torch.softmax(relevant_logits, dim=1)
                current_prediction = hid_prob.argmax(dim=1)   # 映射回原始类别
                
                # 直接使用当前模型的预测结果，不再使用早退出机制
                results = hid_prob

            # 处理所有未早退出的样本（现在不再有早退出机制，所以直接保存所有结果）
            pred = results.argmax(dim=1) + class_offset
            correct += (pred == data.y).sum()

        acc = int(correct) / total_cnt
        return acc


def efficient_eval(test_loader, xs, model_list, prog_list, alpha_list, exit_list, task_id):
    correct = 0
    total_cnt = 0
    for data in test_loader:
        batch_size = data.batch.max().item() + 1
        total_cnt += batch_size
        data = data.to(device)
        
        last = None
        num_classes = len(data.y.unique())
        class_offset = data.y.unique()[0]  # 当前任务的类别起始值
        results = torch.zeros(batch_size, num_classes, device=device)
        not_visited = torch.ones(batch_size, device=device).bool()  # 标记未早退出的样本
        last_prediction = torch.ones(batch_size, device=device) * -1  # 初始化为 -1
        for i, m in enumerate(model_list):
            m.eval()
            prog_list[i].eval()
            exit_list[i].eval()
            alpha_list[i].requires_grad=False
            # 提取当前层的嵌入
            x = prog_list[i]((xs[i][data.original_idx.cpu()]).to(device))
            # 非第 0 层进行加权融合
            if i != 0:
                a = torch.sigmoid(alpha_list[i] / config.T)
                x = x * a + last * (1 - a)

            # 根据 config.encoder 判断是否需要返回 gates
            if config.encoder == 'MOE':
                out, gates = m(x, data.edge_index, return_gates=True)
                # 此处可以选择对 gates 进行进一步处理或收集
                # 例如: gates_list.append(gates)
            else:
                out = m(x, data.edge_index)
            last = out
            hid_out = torch.cat([last[data.root_n_index], global_mean_pool(last, data.batch)], dim=1)
            hid_logits = exit_list[i](hid_out)

            relevant_logits = hid_logits[:, :num_classes]
            hid_prob = torch.softmax(relevant_logits, dim=1)
            current_prediction = hid_prob.argmax(dim=1)   # 映射回原始类别
            # 动态早退出基于熵
            early_mask = ((current_prediction == last_prediction) == not_visited)
            results[early_mask] = hid_prob[early_mask]
            not_visited[early_mask] = False  # 标记已退出样本
            last_prediction = current_prediction
        # 处理未早退出的样本
        results[not_visited] = hid_prob[not_visited]
        pred = results.argmax(dim=1) + class_offset
        correct += (pred == data.y).sum()
        #if task_id==2:
            #print(pred.unique())
            #print(data.y.unique())
    acc = int(correct) / total_cnt
    #print(f'Accuracy: {acc:.4f}') 
    return  acc


def generate_task_datasets(data, num_classes, num_tasks=3, ignore_label=-1, verbose=True):
    """
    分割数据集为多个任务，每个任务固定分配2个类别，多余类别分配给最后一个任务。
    
    参数:
        data (Data): 原始数据集。
        num_classes (int): 类别总数。
        num_tasks (int): 任务数量。
        ignore_label (int): 用于屏蔽非当前任务的标签值。
        verbose (bool): 是否打印统计信息。
        
    返回:
        task_datasets (list of Data): 分割后的任务数据集列表。
        task_class_splits (list of list of int): 每个任务的类别列表。
    """
    # 输入校验
    assert num_tasks >= 1, "任务数必须≥1"
    assert num_classes >= 2, "类别数必须≥2（每个任务至少2个类别）"
    
    # 计算每个任务的类别分配（前n-1个任务各2类，最后一个任务拿剩余类别）
    classes_per_task = 2
    task_class_splits = []
    
    remaining_classes = num_classes
    for t in range(num_tasks):
        if t == num_tasks - 1:  # 最后一个任务拿剩余所有类别
            assigned_classes = remaining_classes
        else:
            assigned_classes = min(classes_per_task, remaining_classes)
        
        if assigned_classes <= 0:
            break  # 无剩余类别可分配
        
        start = num_classes - remaining_classes
        end = start + assigned_classes
        task_class_splits.append(list(range(start, end)))
        remaining_classes -= assigned_classes

    # 分割数据集
    task_datasets = []
    for task_id, classes in enumerate(task_class_splits):
        # 创建任务掩码
        task_mask = torch.zeros_like(data.y, dtype=torch.bool)
        for c in classes:
            task_mask = task_mask | (data.y == c)
        
        # 克隆数据并更新标签和掩码
        task_data = data.clone()
        task_data.y = data.y.clone()
        task_data.y[~task_mask] = ignore_label
        
        # 更新掩码（仅保留当前任务的样本）
        for mask_type in ["train_mask", "val_mask", "test_mask"]:
            if hasattr(data, mask_type):
                setattr(task_data, mask_type, getattr(data, mask_type) & task_mask)
        
        task_datasets.append(task_data)
    
    # 打印统计信息
    if verbose:
        print("\n" + "="*30 + " 任务类别分配统计 " + "="*30)
        print(f"总类别数: {num_classes}, 任务数: {len(task_class_splits)}")
        for task_id, classes in enumerate(task_class_splits):
            print(f"Task {task_id}: 类别 {classes} (共{len(classes)}个类别)")
        
        print("\n" + "="*30 + " 数据集样本统计 " + "="*30)
        for task_id, task_data in enumerate(task_datasets):
            print(f"\n=== Task {task_id} ===")
            valid_mask = task_data.y != ignore_label
            print(f"有效样本数: {int(valid_mask.sum())}")
            
            if hasattr(task_data, "train_mask"):
                print(f"训练集样本: {int(task_data.train_mask.sum())}")
                print(f"验证集样本: {int(task_data.val_mask.sum())}")
                print(f"测试集样本: {int(task_data.test_mask.sum())}")

    return task_datasets, task_class_splits

def split_valid(task_datasets,task_class_splits):
    # 验证 task_data 的相关属性是否正确分离
    for task_id, task_data in enumerate(task_datasets):
        # 获取数据加载器
        train_loader, val_loader, test_loader = get_dataloader(task_data, config, task_id)
        num_val_batches = len(val_loader)  # 如果 val_loader 支持 len() 操作
        #print(f"Task {task_id} - Number of validation batches: {num_val_batches}")


        # 检查 train_loader 中的标签是否只包含当前任务的类别或被屏蔽的标签
        train_labels = []
        for batch in train_loader:
            train_labels.append(batch.y)
        train_labels = torch.cat(train_labels)

        assert torch.all(
            ((train_labels >= task_class_splits[task_id][0]) & (train_labels <= task_class_splits[task_id][-1])) |
            (train_labels == -1)
        ), f"Task {task_id} train_labels 包含无效标签"

        # 检查 val_loader 中的标签
        val_labels = []
        for batch in val_loader:
            val_labels.append(batch.y)
        val_labels = torch.cat(val_labels)
        assert torch.all(
            ((val_labels >= task_class_splits[task_id][0]) & (val_labels <= task_class_splits[task_id][-1])) |
            (val_labels == -1)
        ), f"Task {task_id} val_labels 包含无效标签"

        # 检查 test_loader 中的标签
        test_labels = []
        for batch in test_loader:
            test_labels.append(batch.y)
        test_labels = torch.cat(test_labels)
        assert torch.all(
            ((test_labels >= task_class_splits[task_id][0]) & (test_labels <= task_class_splits[task_id][-1])) |
            (test_labels == -1)
        ), f"Task {task_id} test_labels 包含无效标签"

        print(f"Task {task_id} data verified successfully.")


def expand_linear_layer(linear_layer: torch.nn.Linear, num_new_classes: int):
    r"""
    将 linear_layer (out_features = 原来的 num_classes) 扩充到 (num_classes + num_new_classes)。
    复制老权重到新层，并对新增加的部分进行初始化。
    返回一个新的 nn.Linear。
    """
    in_features = linear_layer.in_features
    old_out_features = linear_layer.out_features
    new_out_features = old_out_features + num_new_classes
    
    # 1) 新建一个更大 out_features 的层
    new_linear = torch.nn.Linear(in_features, new_out_features)
    
    # 2) 复制旧的权重、偏置
    with torch.no_grad():
        new_linear.weight[:old_out_features] = linear_layer.weight
        new_linear.bias[:old_out_features] = linear_layer.bias
    
        # 3) 对新增的部分进行随机初始化
        torch.nn.init.kaiming_uniform_(new_linear.weight[old_out_features:], a=math.sqrt(5))
        # 偏置也可以初始化为 0
        new_linear.bias[old_out_features:].zero_()
    new_linear = new_linear.to(linear_layer.weight.device)
    
    return new_linear

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
    aa_list = []
    af_list = []
    

    for i, seed in enumerate(config.seeds):
        print(f'-------------------------seed {seed}-------------------------------')
        set_seed(seed)
        # load data
        data, text, num_classes = load_data(config.dataset, use_text=True, seed=seed)
        if config.dataset == 'ogbn-products':
            edge_index, _ = to_edge_index(data.edge_index)
            data.edge_index = edge_index
        
        r=config.r # used for dimensionality reduction
        input_dim=config.input_dim # 4096
        k = int(input_dim/r)
        hidden = config.hidden_size
        layer_select = config.layer_select

        # accuracies[k][j] 表示在完成第 k 个任务后对第 j 个任务的测试准确率
        # 索引从0开始，因此当完成第k个任务时实际上是task_id = k
        num_tasks = num_classes // 2
        print(f"num_classes:{num_classes},num_tasks:{num_tasks}")
        accuracies = [[0.0 for _ in range(num_tasks)] for _ in range(num_tasks)]
        AA_list = []  # 存放每个任务完成后的AA值
        AF_list = []  # 存放每个任务完成后的AF值

        task_datasets,task_class_splits = generate_task_datasets (data,num_classes,num_tasks = num_tasks)
        split_valid(task_datasets,task_class_splits)

        encoders = {
            'GCN_Encoder': GCN_Encoder, 
            'GAT_Encoder': GAT_Encoder, 
            'SAGE_Encoder': SAGE_Encoder, 
            'MLP_Encoder': MLP_Encoder,
            'MOE' : MOE ,
            'SparseMOE' : SparseMOE ,
        }
        config.encoder='SparseMOE'
        model_list = [encoders[config.encoder](k, config.layer_num, hidden, k, activation=config.activation, norm=config.norm, last_activation=(l !=len(layer_select)-1), dropout=config.dropout).to(device) for l in layer_select]
        prog_list = [torch.nn.Sequential(torch.nn.Linear(input_dim, k), torch.nn.LayerNorm(k), torch.nn.ReLU(), torch.nn.Linear(k,k)).to(device) for l in layer_select]
        alpha_list = [torch.nn.Parameter(torch.tensor(0.0), requires_grad=True) for l in layer_select]
        exit_list = [torch.nn.Linear(k*2, len(task_class_splits[0])).to(device) for l in layer_select]
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
        
        optimizer = torch.optim.AdamW(params)


        task_acc=[]
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        for task_id, task_data in enumerate(task_datasets):
            print(f"\n--- Training on Task {task_id} ---")
            train_loader, val_loader, test_loader = get_dataloader(task_data, config, task_id)

            new_num_classes = len(task_class_splits[task_id])
            if task_id > 0:
                for i in range(len(exit_list)):
                    # 找到并移除对应的旧的参数组
                    for param_group in optimizer.param_groups:
                        if param_group['params'] == list(exit_list[i].parameters()):
                            optimizer.param_groups.remove(param_group)
                            break
                    exit_list[i] = expand_linear_layer(exit_list[i], new_num_classes)       # 更新 exit_list[i] 为新的输出层
                    optimizer.add_param_group({'params': exit_list[i].parameters(), 'lr': lr, 'weight_decay': weight_decay})
                    

            acc = efficient_train_eval(train_loader, val_loader, test_loader, xs_list, model_list, prog_list, alpha_list, exit_list, optimizer,loss_fn,task_id,task_class_splits)            
            current_task_acc = efficient_eval(test_loader, xs_list, model_list, prog_list, alpha_list, exit_list,task_id)

            # 保存当前任务测试集的准确率
            task_acc.append(acc)
            print(f"Task {task_id} Accuracy: {acc:.4f} , current_task_acc:{current_task_acc}")
           

            # 对已完成的所有任务重新评估来计算AA和AF
            all_tasks_acc = []
            for prev_task_id in range(task_id + 1):
                # 使用之前任务的数据集的测试集来评估
                _, _, prev_test_loader = get_dataloader(task_datasets[prev_task_id], config, task_id)
                prev_task_acc = efficient_eval(prev_test_loader, xs_list, model_list, prog_list, alpha_list, exit_list,task_id)
                accuracies[task_id][prev_task_id] = prev_task_acc
                all_tasks_acc.append(prev_task_acc)
                print(f"prev_task_id: {prev_task_id},prev_task_acc:{prev_task_acc}")
            # 计算AA_k (task_id从0计数，所以加1)
            AA_k = sum(all_tasks_acc) / (task_id+1)
            AA_list.append(AA_k)

            
            AF_k=0
            # 计算AF (Average Forgetting)
            if task_id > 0:
                total_forgetting = 0
                for prev_task_id in range(task_id):
                    forgetting = accuracies[task_id][prev_task_id] - accuracies[prev_task_id][prev_task_id]
                    total_forgetting += forgetting
                AF_k = total_forgetting / task_id  # 计算AF
                AF_list.append(AF_k)
            
            # 输出每个任务训练后的AA和AF
            print(f"After training task {task_id}: AA = {AA_k:.4f}, AF_k = {AF_k:.4f}")

        # 输出最终的AA和AF
        final_AA = AA_list[-1] if AA_list else 0
        final_AF = AF_list[-1] if AF_list else 0

        print(f"\nFinal AA: {final_AA:.4f}")
        print(f"Final AF: {final_AF:.4f}")
        # 输出总体性能
        #acc, acc_std = np.mean(task_acc), np.std(task_acc)
        #print(f"# seed{seed}_acc: {acc*100:.2f}±{acc_std*100:.2f}")
        aa_list.append(final_AA)
        af_list.append(final_AF)
    print("\n=== Final Results ===")
    print(f"# AA: {np.mean(aa_list)*100:.2f}±{np.std(aa_list)*100:.2f}")
    print(f"# AF: {np.mean(af_list)*100:.2f}±{np.std(af_list)*100:.2f}")
    
    
    
