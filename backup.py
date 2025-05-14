def generate_task_datasets(data, num_classes, num_tasks=2, ignore_label=-1):
    """
    分割数据集为多个任务，并屏蔽非当前任务的标签。
    
    参数:
        data (Data): 原始数据集。
        num_classes (int): 类别总数。
        num_tasks (int): 任务数量。
        ignore_label (int): 用于屏蔽非当前任务的标签值。
        
    返回:
        task_datasets (list of Data): 分割后的任务数据集列表。
        task_class_splits (list of list of int): 每个任务的类别列表。
    """
    # 获取样本总数
    total_samples = data.y.size(0)
    print(f"数据集样本总数: {total_samples}")
    
    # 统计每个类别（包括忽略标签）的样本数
    unique_labels, counts = torch.unique(data.y, return_counts=True)
    
    print("各类别样本数统计:")
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        if label == ignore_label:
            print(f"  忽略标签 ({ignore_label}) 的样本数: {count}")
        else:
            print(f"  类别 {label} 的样本数: {count}")
    # 第一个任务占总类别的一半
    first_task_classes = num_classes // 2
    remaining_classes = num_classes - first_task_classes
    remaining_tasks = num_tasks - 1
    # 其余任务均分
    classes_per_task = remaining_classes // remaining_tasks
    extra_classes = remaining_classes % remaining_tasks  # 若有剩余类别
    task_class_splits = []

    # 第一个任务的类别
    start = 0
    end = start + first_task_classes
    task_class_splits.append(list(range(start, end)))
    
    # 后续任务的类别
    start = end
    for t in range(remaining_tasks):
        end = start + classes_per_task + (1 if t < extra_classes else 0)
        task_class_splits.append(list(range(start, end)))
        start = end

    task_datasets = []
    for task_id, classes in enumerate(task_class_splits):
        # 创建任务掩码
        task_mask = torch.zeros_like(data.y, dtype=torch.bool)
        for c in classes:
            task_mask = task_mask | (data.y == c)

        # 克隆数据并更新掩码
        task_data = data.clone()
        task_data.train_mask = data.train_mask & task_mask
        task_data.val_mask = data.val_mask & task_mask
        task_data.test_mask = data.test_mask & task_mask

        # 屏蔽非当前任务的标签
        #print("Task 0 mask true count:", task_mask.sum())
        #print("Unique labels before masking:", task_data.y.unique())
        task_data.y = data.y.clone()
        task_data.y[~task_mask] = ignore_label  # 设置为忽略标签
        #print("Labels after masking:", task_data.y.unique())

        task_datasets.append(task_data)

    print("Task Class Splits:")
    for task_id, (data, classes) in enumerate(zip(task_datasets, task_class_splits)):

        print(f"========== Task {task_id} ==========")
        
        # 打印任务数据集的样本总数
        total_samples = data.y.size(0)
        print(f"数据集样本总数: {total_samples}")
        
        # 如果数据集包含训练、验证和测试掩码，则分别统计各部分的样本数
        if hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(data, "test_mask"):
            train_count = int(data.train_mask.sum())
            val_count = int(data.val_mask.sum())
            test_count = int(data.test_mask.sum())
            print(f"训练集样本数: {train_count}")
            print(f"验证集样本数: {val_count}")
            print(f"测试集样本数: {test_count}")
            print(f"任务标签  : {classes}")
        
        # 只统计有效标签的样本数量（过滤掉屏蔽标签 ignore_label）
        valid_mask = data.y != ignore_label
        if valid_mask.sum() == 0:
            print("该任务没有有效标签的样本。")
        else:
            valid_labels = data.y[valid_mask]
            unique_labels, counts = torch.unique(valid_labels, return_counts=True)
            #for label, count in zip(unique_labels.tolist(), counts.tolist()):
                #print(f"类别 {label} 的样本数量: {count}")
        print("\n")


    return task_datasets, task_class_splits



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
    print(num_classes, num_tasks
    )
    
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
    print("任务划分类别:", task_class_splits)

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
