import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import math

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载MNIST训练和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 分割数据集为任务1（0-4）和任务2（5-9）
def split_dataset(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)

# 定义任务1和任务2的类别
task1_classes = [0, 1, 2, 3, 4]
task2_classes = [5, 6, 7, 8, 9]

# 创建任务1和任务2的训练和测试集
train_task1 = split_dataset(train_dataset, task1_classes)
train_task2 = split_dataset(train_dataset, task2_classes)

test_task1 = split_dataset(test_dataset, task1_classes)
test_task2 = split_dataset(test_dataset, task2_classes)

# 创建数据加载器
batch_size = 64
train_loader_task1 = DataLoader(train_task1, batch_size=batch_size, shuffle=True)
train_loader_task2 = DataLoader(train_task2, batch_size=batch_size, shuffle=True)

test_loader_task1 = DataLoader(test_task1, batch_size=batch_size, shuffle=False)
test_loader_task2 = DataLoader(test_task2, batch_size=batch_size, shuffle=False)

# 定义模型
class IncrementalModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, initial_num_classes=5):
        super(IncrementalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, initial_num_classes)  # 初始任务1有5个类别

    def add_classes(self, new_num_classes):
        # 扩展输出层以包含新类别
        in_features = self.fc2.in_features
        out_features = self.fc2.out_features
        new_fc2 = nn.Linear(in_features, out_features + new_num_classes)
        
        # 初始化新层的权重
        nn.init.kaiming_uniform_(new_fc2.weight, a=math.sqrt(5))
        new_fc2.bias.data.zero_()
        
        # 复制旧层的权重到新层
        new_fc2.weight.data[:out_features] = self.fc2.weight.data
        new_fc2.bias.data[:out_features] = self.fc2.bias.data
        
        self.fc2 = new_fc2

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 训练函数
def train(model, dataloader, criterion, optimizer, epochs=5, task_id=1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            
            # 对任务2，标签需要映射到0-4
            if task_id == 2:
                labels = labels - 5
            
            optimizer.zero_grad()
            outputs = model(data)
            if task_id == 1:
                loss = criterion(outputs, labels)
            elif task_id == 2:
                relevant_outputs = outputs[:, 5:]  # 任务2的输出
                loss = criterion(relevant_outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# 测试函数
def test(model, dataloader, task_id, task_classes):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            
            # 打印当前批次的标签和预测
            outputs = model(data)
            if task_id == 1:
                relevant_outputs = outputs[:, :5]  # 任务1有5个类别
                relevant_labels = labels
                _, predicted = torch.max(relevant_outputs, 1)
                # 预测的类别不需要调整
            elif task_id == 2:
                relevant_outputs = outputs[:, 5:]  # 任务2有5个新类别
                relevant_labels = labels - 5
                _, predicted = torch.max(relevant_outputs, 1)
                # 将预测结果映射回原始类别范围（5-9）
                predicted = predicted + 5
            else:
                raise ValueError("Unsupported task ID")
            
            if batch_idx<5:
                # 打印当前批次的标签和预测结果
                print(f"Task {task_id} - Batch {batch_idx+1}:")
                print(f"  Labels:     {labels.cpu().numpy()}")
                print(f"  Predictions:{predicted.cpu().numpy()}")
            
            # 计算准确率
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f"Task {task_id} Accuracy: {accuracy:.2f}%\n")

# 初始化模型，任务1有5个类别
model = IncrementalModel().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("训练任务1 (类别 0-4)")
train(model, train_loader_task1, criterion, optimizer, epochs=5, task_id=1)

print("\n测试任务1")
test(model, test_loader_task1, task_id=1, task_classes=task1_classes)

# 扩展模型以包含任务2的5个新类别
model.add_classes(new_num_classes=5)
model.to(device)

# 定义新的优化器（优化所有参数）
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n训练任务2 (类别 5-9)")
train(model, train_loader_task2, criterion, optimizer, epochs=5, task_id=2)

print("\n测试任务1")
test(model, test_loader_task1, task_id=1, task_classes=task1_classes)

print("\n测试任务2")
test(model, test_loader_task2, task_id=2, task_classes=task2_classes)
