import json
import os
import glob
import random  # 新增：用于固定 Python 原生随机种子
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- 1. 新增：固定随机种子函数 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多 GPU
    # 保证 CuDNN 的操作也是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置环境变量以确保某些特定的 CUDA 操作也是确定的
    os.environ['PYTHONHASHSEED'] = str(seed)


# --- 配置参数 ---
T_WINDOW = 80
N_POINTS = 10  # 建议根据需求设为 128 或保持 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
X_MIN = -2
X_MAX = 2
Y_MIN = -2.5
Y_MAX = 0
class RadarDataset(Dataset):
    def __init__(self, file_paths, mode='train', T=64, N=128):
        self.features = []
        self.labels = []
        self.metadata = [] # 用于存储文件名和原始坐标
        self.T = T
        self.N = N
        
        for fp in file_paths:
            self._process_json(fp, mode)

    def _process_json(self, file_path, mode):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
        all_frames = content.get('data', [])
        total_f = len(all_frames)
        if total_f < self.T:
            return 
            
        mid = total_f // 2
        if mode == 'train':
            frames = all_frames[:mid]
        else:
            frames = all_frames[mid:]
            
        filename = os.path.basename(file_path)
        label = 1 if "EXT" in filename.upper() else 0
        
        for i in range(len(frames) - self.T + 1):
            window_frames = frames[i : i + self.T]
            combined_points = []
            for frame in window_frames:
                raw_points = frame.get('points', [])
                
                # --- 新增：坐标范围过滤 ---
                # 过滤规则：x 轴必须在 [-2, 2] 范围内，y 轴必须在 [-2.5, 0] 范围内
                filtered_points = [
                    p for p in raw_points 
                    if X_MIN <= p['pos'][0] <= X_MAX and Y_MIN <= p['pos'][1] <= Y_MAX
                ]
                combined_points.extend(filtered_points)
            
            # 先过滤再按照 SNR 排序并取前 N 个
            combined_points.sort(key=lambda x: x.get('snr', 0), reverse=True)
            top_points = combined_points[:self.N]
            
            # 构造 (N, 2) 特征
            feature_vec = np.zeros((self.N, 2), dtype=np.float32)
            for j, p in enumerate(top_points):
                feature_vec[j] = p['pos']
            
            self.features.append(feature_vec.flatten())
            self.labels.append(label)
            # 保存元数据以便后续追溯
            self.metadata.append({
                "source_file": filename,
                "raw_points": feature_vec.tolist(),
                "window_idx": i
            })

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.long),
                idx)

# --- 神经网络定义 ---
class RadarMLP(nn.Module):
    def __init__(self, input_size):
        super(RadarMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.net(x)

# --- 训练与评估逻辑 ---
def train_model(data_dir):
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print(f"错误: 路径 '{data_dir}' 下没有找到 JSON 文件。")
        return

    train_ds = RadarDataset(json_files, mode='train', T=T_WINDOW, N=N_POINTS)
    test_ds = RadarDataset(json_files, mode='test', T=T_WINDOW, N=N_POINTS)
    
    if len(train_ds) == 0 or len(test_ds) == 0:
        print("错误: 经过过滤后数据集为空。")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"训练集样本数: {len(train_ds)}, 测试集样本数: {len(test_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarMLP(input_size=N_POINTS * 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 关键修改：初始化最优指标 ---
    best_acc = -1.0
    error_log = [] 

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        model.eval()
        correct = 0
        total = 0
        current_epoch_errors = []
        
        with torch.no_grad():
            for inputs, targets, indices in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                # 记录当前 batch 的错误
                mask = (predicted != targets)
                error_indices = mask.nonzero(as_tuple=True)[0]
                
                for idx_in_batch in error_indices:
                    global_idx = indices[idx_in_batch].item()
                    meta = test_ds.metadata[global_idx]
                    
                    error_info = {
                        "epoch": epoch + 1,
                        "file": meta["source_file"],
                        "true_label": "EXT" if targets[idx_in_batch].item() == 1 else "INT",
                        "pred_label": "EXT" if predicted[idx_in_batch].item() == 1 else "INT",
                        "points": meta["raw_points"]
                    }
                    current_epoch_errors.append(error_info)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        acc = 100 * correct / total if total > 0 else 0
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}%")
        
        # --- 关键修改：如果当前准确率是最高的，则保存错误日志 ---
        if acc >= best_acc:
            best_acc = acc
            error_log = current_epoch_errors
            # 可选：这里也可以保存模型权重 torch.save(model.state_dict(), 'best_model.pth')

    if error_log:
        save_path = "classification_errors.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            # 在保存的数据中可以加入最高准确率的信息以便参考
            json.dump({
                "best_acc": f"{best_acc:.2f}%",
                "errors": error_log
            }, f, indent=4, ensure_ascii=False)
        print(f"\n[完成] 准确率最高轮次的准确率为 {best_acc:.2f}%，已保存其对应的 {len(error_log)} 条错误数据至: {save_path}")
    else:
        print("\n验证通过：未发现分类错误。")

if __name__ == "__main__":
    # 新增：固定随机种子
    set_seed(42)
        
    # 请根据您的实际路径修改
    target_path = r"D:\uwb occupy data collection\数据1229"
    if os.path.isdir(target_path):
        train_model(target_path)
    else:
        print(f"路径无效: {target_path}")