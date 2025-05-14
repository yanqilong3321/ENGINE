# task_vector.py
import torch
import torch.nn as nn

class TaskVector:
    def __init__(self, pretrained_backbone_state_dict=None, finetuned_backbone_state_dict=None, vector=None):
        if vector is not None:
            self.vector = vector
        else:
            self.vector = {}
            for key in pretrained_backbone_state_dict:
                if key.startswith('conv1') or key.startswith('conv2'):
                    if key in finetuned_backbone_state_dict and pretrained_backbone_state_dict[key].size() == finetuned_backbone_state_dict[key].size():
                        self.vector[key] = finetuned_backbone_state_dict[key] - pretrained_backbone_state_dict[key]
                    else:
                        print(f'警告: 键 {key} 在微调模型中不存在或尺寸不匹配。')

    def __add__(self, other):
        """将两个任务向量相加。"""
        new_vector = {}
        for key in self.vector:
            if key in other.vector:
                new_vector[key] = self.vector[key] + other.vector[key]
            else:
                print(f'警告: 键 {key} 不存在于两个任务向量中。')
        return TaskVector(vector=new_vector)

    def __neg__(self):
        """将任务向量取反。"""
        new_vector = {key: -value for key, value in self.vector.items()}
        return TaskVector(vector=new_vector)

    def apply_to(self, backbone_class, pretrained_checkpoint, scaling_coef=1.0):
        """
        Apply a task vector to a pretrained backbone model and return a new modified model.
        
        Args:
            backbone_class (callable): Callable that returns a new backbone instance.
            pretrained_checkpoint (str): Path to the pretrained backbone state dict.
            scaling_coef (float): Scaling coefficient for the task vector.
        
        Returns:
            nn.Module: New backbone model with applied task vector.
        """
        # 实例化一个新的 Backbone 模型
        new_backbone = backbone_class()
        
        # 加载预训练的 Backbone 状态字典
        pretrained_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
        new_backbone.load_state_dict(pretrained_state_dict, strict=False)
        
        # 应用任务向量到新的 Backbone
        with torch.no_grad():
            backbone_state = new_backbone.state_dict()
            for key in self.vector:
                if key in backbone_state:
                    backbone_state[key] += scaling_coef * self.vector[key]
                else:
                    print(f'警告: 键 {key} 不存在于 Backbone 中，保持原值。')
            new_backbone.load_state_dict(backbone_state)
        
        return new_backbone

def load_task_vector(path):
    vector = torch.load(path, map_location='cpu')
    return TaskVector(vector=vector)
