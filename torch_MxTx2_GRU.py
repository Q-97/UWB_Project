import json
import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- 配置参数 ---
T_WINDOW = 80
M_POINTS = 4
BATCH_SIZE = 32
LEARNING_RATE = 0.001/3
EPOCHS = 40
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX = -2.5, 0
# --- 新增 Loss 配置 ---
# 可选: "CE" 或 "Focal"
LOSS_TYPE = "CE" 
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
HIDDEN_SIZE = 64
LAYER_NUM = 2

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C], targets: [N]
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class RadarDataset(Dataset):
    def __init__(self, file_paths, mode='train', T=64, M=10):
        self.features = []
        self.labels = []
        self.metadata = [] 
        self.T = T
        self.M = M
        for fp in file_paths:
            self._process_json(fp, mode)

    def _process_json(self, file_path, mode):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        all_frames = content.get('data', [])
        total_f = len(all_frames)
        if total_f < self.T: return 
        mid = total_f // 2
        frames = all_frames[:mid] if mode == 'train' else all_frames[mid:]
        filename = os.path.basename(file_path)
        label = 1 if "EXT" in filename.upper() else 0
        
        for i in range(len(frames) - self.T + 1):
            window_frames = frames[i : i + self.T]
            window_feature = np.zeros((self.T, self.M, 2), dtype=np.float32)
            for t_idx, frame in enumerate(window_frames):
                raw_points = frame.get('points', [])
                filtered = [p for p in raw_points if X_MIN <= p['pos'][0] <= X_MAX and Y_MIN <= p['pos'][1] <= Y_MAX]
                filtered.sort(key=lambda x: x.get('snr', 0), reverse=True)
                for m_idx, p in enumerate(filtered[:self.M]):
                    window_feature[t_idx, m_idx] = p['pos']
            
            self.features.append(window_feature.flatten())
            self.labels.append(label)
            self.metadata.append({
                "source_file": filename,
                "window_idx": i,
                "points": window_feature.tolist() 
            })

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        # 原代码：self.features[idx] 是展平的
        # 修改：将其 reshape 为 (时间步 T, 特征数 M*2)
        feature = torch.tensor(self.features[idx]).view(self.T, -1) 
        return feature, torch.tensor(self.labels[idx]), idx

class RadarMLP(nn.Module):
    def __init__(self, input_size):
        super(RadarMLP, self).__init__()
        # 定义隐藏层比例，可以根据需要调整这个基数 (e.g., 64)
        base = 64 
        
        self.net = nn.Sequential(
            # 第一层最消耗参数，控制在 base 左右
            nn.Linear(input_size, base),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 中间层减半
            nn.Linear(base, base // 2),
            nn.ReLU(),
            
            # 输出层
            nn.Linear(base // 2, 2)
        )

    def forward(self, x):
        return self.net(x)

class RadarGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(RadarGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 层
        # batch_first=True 表示输入形状为 (Batch, Seq, Feature)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # 分类输出层
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x 形状: (Batch, T, M*2)
        
        # out: (Batch, T, hidden_size) 包含每个时间步的输出
        # _ : 隐藏状态
        out, _ = self.gru(x)
        
        # 我们只取最后一个时间步 (T_WINDOW) 的输出进行分类
        last_time_step_out = out[:, -1, :] 
        
        logits = self.fc(last_time_step_out)
        return logits

def train_one_seed(data_dir, seed, save_dir):
    """进行单一随机数下的模型训练"""
    set_seed(seed)
    print(f"\n>>> 开始训练 - Seed: {seed}")
    
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    train_ds = RadarDataset(json_files, mode='train', T=T_WINDOW, M=M_POINTS)
    test_ds = RadarDataset(json_files, mode='test', T=T_WINDOW, M=M_POINTS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"训练集样本数: {len(train_ds)}, 测试集样本数: {len(test_ds)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_size = M_POINTS * 2 (每个时间点有 4 个点 * 2 个坐标 = 8 个特征)
    model = RadarGRU(input_size=M_POINTS * 2, hidden_size=HIDDEN_SIZE, num_layers=LAYER_NUM).to(device)
    # --- 修改 Loss 选择逻辑 ---
    if LOSS_TYPE == "Focal":
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = -1.0
    best_epoch = -1
    best_error_log = []

    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct, total, current_errors = 0, 0, []
        with torch.no_grad():
            for inputs, targets, indices in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                
                mask = (pred != targets)
                for idx_in_batch in mask.nonzero(as_tuple=True)[0]:
                    meta = test_ds.metadata[indices[idx_in_batch]]
                    current_errors.append({
                        "epoch": epoch + 1,
                        "file": meta["source_file"],
                        "window_idx": meta["window_idx"],
                        "true_label": "EXT" if targets[idx_in_batch] == 1 else "INT",
                        "pred_label": "EXT" if pred[idx_in_batch] == 1 else "INT",
                        "points": meta["points"]
                    })
                total += targets.size(0)
                correct += (pred == targets).sum().item()
        
        acc = 100 * correct / total
        if acc >= best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_error_log = current_errors
        
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Seed {seed} | Epoch [{epoch+1}/{EPOCHS}], Test Acc: {acc:.2f}%")

    # 保存该 seed 下的结果
    save_name = f"errors_seed{seed}_acc{best_acc:.2f}.json"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'w') as f:
        json.dump({
            "seed": seed,
            "best_acc": f"{best_acc:.2f}%", 
            "best_epoch": best_epoch,
            "errors": best_error_log
        }, f, indent=4)
    
    print(f"[Seed {seed} 完成] 最佳准确率: {best_acc:.2f}% (来自 Epoch {best_epoch})")
    return best_acc

if __name__ == "__main__":
    # 1. 准备路径和文件夹
    target_data_path = r"D:\uwb occupy data collection\数据1229"
    
    # --- 修改文件夹命名规则 ---
    model_type = "GRU"
    loss_str = f"{LOSS_TYPE}"
    if LOSS_TYPE == "Focal":
        loss_str += f"_G{FOCAL_GAMMA}_A{FOCAL_ALPHA}"
        
    folder_name = f"Res_{model_type}_{loss_str}_T{T_WINDOW}_M{M_POINTS}_B{BATCH_SIZE}_LR{LEARNING_RATE:.5f}_E{EPOCHS}_X{X_MIN}_{X_MAX}_Y{Y_MIN}_{Y_MAX}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"创建结果文件夹: {folder_name}")
    # 2. 定义三个随机数
    seeds = [42, 123, 2025]
    all_best_accs = []

    # 3. 循环训练
    for s in seeds:
        max_acc = train_one_seed(target_data_path, s, folder_name)
        all_best_accs.append(max_acc)

    # 4. 计算并打印最终统计结果
    avg_max_acc = sum(all_best_accs) / len(all_best_accs)
    print("\n" + "="*50)
    print(f"所有实验完成！")
    print(f"文件夹位置: {os.path.abspath(folder_name)}")
    print(f"各随机数下的最佳准确率: {all_best_accs}")
    print(f"三个随机数下的平均最大准确率: {avg_max_acc:.2f}%")
    print("="*50)