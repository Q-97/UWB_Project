import json
import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd # 用于生成汇总表格

import sys
import argparse

# --- 使用 argparse 替代硬编码参数 ---
parser = argparse.ArgumentParser()
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--data_path', type=str, default=r"/home/zenshangyou/radar_point_cloud_samples_1x4/数据0105 降采样")
args = parser.parse_args()

# 将解析到的值赋给原有的变量名



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- 参数配置（基础值） ---
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX = -2.5, 0
LOSS_TYPE = "CE" 
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
WINDOW_SIZE = 64
DROPOUT = args.dropout
target_data_path_cmd = args.data_path
# --- 逻辑类保持不变 ---

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

class RadarDataset(Dataset):
    def __init__(self, file_paths, mode='train', window_size=128, T=64, M=10, norm_type=0):
        self.features, self.labels, self.metadata = [], [], []
        self.window_size, self.T, self.M, self.norm_type = window_size, T, M, norm_type
        for fp in file_paths:
            self._process_json(fp, mode)

    def _process_json(self, file_path, mode):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        all_frames = content.get('data', [])
        total_f = len(all_frames)
        
        # 必须保证总帧数大于等于我们定义的滑窗大小 128
        if total_f < self.window_size: return 
        
        mid = total_f // 2
        frames = all_frames[:mid] if mode == 'train' else all_frames[mid:]
        filename = os.path.basename(file_path)
        label = 1 if "EXT" in filename.upper() else 0
        
        # 计算下采样的索引序列
        # 例如 window_size=128, T=32，则每隔4帧取一帧
        sample_indices = np.linspace(0, self.window_size - 1, self.T, dtype=int)

        for i in range(len(frames) - self.window_size + 1):
            # 1. 先取出原始的 128 帧窗口
            raw_window = frames[i : i + self.window_size]
            
            # 2. 进行下采样抽取 T 帧
            window_feature = np.zeros((self.T, self.M, 2), dtype=np.float32)
            for t_idx, original_idx in enumerate(sample_indices):
                frame = raw_window[original_idx]
                raw_points = frame.get('points', [])
                filtered = [p for p in raw_points if X_MIN <= p['pos'][0] <= X_MAX and Y_MIN <= p['pos'][1] <= Y_MAX]
                filtered.sort(key=lambda x: x.get('snr', 0), reverse=True)
                for m_idx, p in enumerate(filtered[:self.M]):
                    window_feature[t_idx, m_idx] = p['pos']
            # --- 应用标准化 (STD Normalization) ---
            normalized_feat = normalize_feature(window_feature, self.norm_type)
            self.features.append(normalized_feat.flatten())
            self.labels.append(label)
            self.metadata.append({
                "source_file": filename,
                "window_idx": i,
                "points": window_feature.tolist() 
            })

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        # 注意：此处返回的 T 是下采样后的 T
        feature = torch.tensor(self.features[idx]).view(self.T, -1) 
        return feature, torch.tensor(self.labels[idx]), idx

# --- 补充 RadarMLP 模型 ---
class RadarMLP(nn.Module):
    def __init__(self, T, M, hidden_dim):
        super(RadarMLP, self).__init__()
        input_size = T * M * 2  # 将时间轴完全展开
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        # x: [Batch, T, M*2] -> [Batch, T*M*2]
        x = x.view(x.size(0), -1)
        return self.net(x)

# --- 新增的 TemporalMLP 类 ---
class TemporalMLP(nn.Module):
    def __init__(self, M, hidden_dim=64):
        """
        M: 每帧的点数
        hidden_dim: 隐藏层维度
        """
        super(TemporalMLP, self).__init__()
        # 1. 空间特征提取：对每一帧的 M*2 个坐标进行独立处理 (权重在时间维度上共享)
        self.spatial_extractor = nn.Sequential(
            nn.Linear(M * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. 分类器：处理池化后的全局特征
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        # x 形状: [Batch, T, M*2]
        # 第一步：应用空间提取器。PyTorch 的 Linear 会自动处理 [B, T, Feature] 的最后一位
        x = self.spatial_extractor(x) # 结果形状: [Batch, T, hidden_dim]
        
        # 第二步：全局时间最大池化 (核心改进)
        # 在时间轴 T (dim=1) 上取最大值，捕捉 96 帧中任何一处最显著的特征
        x, _ = torch.max(x, dim=1) # 结果形状: [Batch, hidden_dim]
        
        # 第三步：分类
        return self.classifier(x)



class UnifiedRadarModel(nn.Module):
    def __init__(self, model_type='temporalmlp', T=128, M=10, hidden_dim=64):
        super(UnifiedRadarModel, self).__init__()
        # 统一转为小写处理
        self.model_type = model_type.lower()
        self.hidden_dim = hidden_dim
        self.T = T
        self.M = M
        input_feature_dim = M * 2

        # 修改这里的判断条件为小写
        if self.model_type == 'temporalmlp':
            # 当 hidden_dim 较小时（如 2, 4, 8, 16），我们把它当作 Embedding 维度
            # 即使 hidden_dim 为 0，我们也给它一个默认极小值 2
            eff_hidden = max(2, hidden_dim)
            if hidden_dim <= 32: # 这里的阈值可以自己定，代表“简单模式”
                if eff_hidden == 2:
                    self.spatial_extractor = nn.Linear(input_feature_dim, 2)

                    self.classifier = nn.Identity()
                else:
                    # 相当于一个简单的线性 Embedding 层
                    # self.spatial_extractor = nn.Linear(input_feature_dim, eff_hidden)
                    # 分类器必须把 Embedding 映射到 2（类别数）
                    if DROPOUT !=0:
                        self.spatial_extractor = nn.Sequential(
                        nn.Linear(input_feature_dim, eff_hidden),
                        nn.Dropout(DROPOUT)
                        #?nn.Linear(hidden_dim, hidden_dim),
                        #nn.ReLU()
                    )
                    else:
                        self.spatial_extractor = nn.Linear(input_feature_dim, eff_hidden)
                    self.classifier = nn.Linear(eff_hidden, 2)
            else:
                self.spatial_extractor = nn.Sequential(
                    nn.Linear(input_feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
                    nn.ReLU(),
                    nn.Linear(max(1, hidden_dim // 2), 2)
                )
        
        elif self.model_type == 'radarmlp':
            flatten_input_dim = T * M * 2
            if hidden_dim == 0:
                self.net = nn.Linear(flatten_input_dim, 2)
            else:
                self.net = nn.Sequential(
                    nn.Linear(flatten_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
                    nn.ReLU(),
                    nn.Linear(max(1, hidden_dim // 2), 2)
                )

    def forward(self, x):
        # 注意：这里的判断也要改为小写
        if self.model_type == 'temporalmlp':
            # x: [Batch, T, M*2]
            x = self.spatial_extractor(x)  # [Batch, T, eff_hidden]
            x, _ = torch.max(x, dim=1)     # 全局时间最大池化 [Batch, eff_hidden]
            x = self.classifier(x)         # [Batch, 2]
            return x
        else: # radarmlp
            x = x.view(x.size(0), -1)
            return self.net(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# --- 修改后的训练入口 ---
def train_one_seed(data_dir, seed, save_dir, model_type, T_down, M, H, LR, BATCH, EPOCH, norm_type):
    set_seed(seed)
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    train_ds = RadarDataset(json_files, mode='train', window_size=WINDOW_SIZE, T=T_down, M=M, norm_type=norm_type)
    test_ds = RadarDataset(json_files, mode='test', window_size=WINDOW_SIZE, T=T_down, M=M, norm_type=norm_type)
    
    if len(train_ds) == 0: return 0.0, 0
    
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UnifiedRadarModel(
        model_type=model_type, # 'temporal' 或 'radar'
        T=T_down, 
        M=M, 
        hidden_dim=H
    ).to(device)

    num_params = count_parameters(model)
    
    if LOSS_TYPE == "Focal":
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = -1.0
    best_epoch = -1
    best_error_log = []

    for epoch in range(EPOCH):
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
        correct, total, current_errors = 0, 0, []
        with torch.no_grad():
            for inputs, targets, indices in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                
                # 错误样本记录逻辑
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
        avg_loss = running_loss / len(train_loader)
        # --- 打印阶段性结果 ---
        # 每 5 个 Epoch 打印一次，或者在首尾 Epoch 打印
        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == EPOCH:
            print(f"      [Epoch {epoch+1:3d}/{EPOCH}] Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
        if acc >= best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_error_log = current_errors

    # 原有的保存 JSON 逻辑
    save_name = f"errors_seed{seed}_acc{best_acc:.2f}.json"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'w') as f:
        json.dump({"seed": seed, "best_acc": f"{best_acc:.2f}%", "best_epoch": best_epoch, "errors": best_error_log}, f, indent=4)
    
    print(f"   [Seed {seed}] Best Acc: {best_acc:.2f}% at Epoch {best_epoch}")
    return best_acc, num_params

def normalize_feature(feat, norm_type):
    """
    feat 形状: (T, M, 2) 其中 2 代表 (x, y)
    norm_type: 
        0: 不做处理
        1: 对每一个样本里面所有点标准化 (计算整个矩阵的 mean 和 std)
        2: 对每一个样本里面x做x的标准化，y做y的标准化 (全局 XY 均值/方差独立)
        3: 对不同T下的M*2个数值统一标准化 (逐帧计算 mean 和 std)
        4: 对不同T下的M个点，x做x的标准化，y做y的标准化 (逐帧 XY 均值/方差独立)
    """
    if norm_type == 0:
        return feat
    
    eps = 1e-8 # 防止除以 0
    feat_norm = feat.copy()

    if norm_type == 1:
        # 1. 每一个样本所有点统一标准化
        mu, std = feat.mean(), feat.std()
        feat_norm = (feat - mu) / (std + eps)

    elif norm_type == 2:
        # 2. 每一个样本里面 x 和 y 独立标准化
        # x_feat: (T, M), y_feat: (T, M)
        for i in range(2):
            mu, std = feat[:, :, i].mean(), feat[:, :, i].std()
            feat_norm[:, :, i] = (feat[:, :, i] - mu) / (std + eps)

    elif norm_type == 3:
        # 3. 每一个样本中，每一帧(T)独立进行标准化
        for t in range(feat.shape[0]):
            mu, std = feat[t].mean(), feat[t].std()
            feat_norm[t] = (feat[t] - mu) / (std + eps)

    elif norm_type == 4:
        # 4. 每一个样本中，每一帧(T)独立进行 XY 独立标准化
        for t in range(feat.shape[0]):
            for i in range(2):
                mu, std = feat[t, :, i].mean(), feat[t, :, i].std()
                feat_norm[t, :, i] = (feat[t, :, i] - mu) / (std + eps)

    return feat_norm


if __name__ == "__main__":
    target_data_path = target_data_path_cmd
    # 1. 建立根保存目录名（基于数据文件夹名称）
    # rstrip 防止路径末尾有斜杠导致获取到空字符串
    data_folder_name = os.path.basename(target_data_path.rstrip(os.sep))
    root_save_dir = f"Results_{data_folder_name}_DO_{DROPOUT}"
    if not os.path.exists(root_save_dir):
        os.makedirs(root_save_dir)


    MODEL_TYPES = ["TemporalMLP"]#, "RadarMLP"
    M_LIST = [2,4,8,16]
    T_DOWNSAMPLE_LIST = [16,32,64] # 下采样后的帧数
    HIDDEN_LIST = [2,4,8,16,32,64]
    
    SEEDS = [42, 123,2025,4,6,8]
    LR = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    # 5种模式逻辑
    NORM_MODES = [0] 
    NORM_NAMES = {
        0: "不进行标准化",
        1: "样本全局标准化",
        2: "样本XY独立标准化",
        3: "逐帧全局标准化",
        4: "逐帧XY独立标准化"
    }
    summary_data = []

    # --- 开始嵌套遍历 ---
    for n_type in NORM_MODES:
        for mtype in MODEL_TYPES:
            for m_val in M_LIST:
                for t_down in T_DOWNSAMPLE_LIST:
                    for h_val in HIDDEN_LIST:
                        # 2. 构建树状结构路径
                        # 结构: Results_数据1229 / Model / Norm_Mode / Params
                        norm_desc = f"Norm{n_type}_{NORM_NAMES[n_type]}"
                        param_desc = f"T{t_down}_M{m_val}_H{h_val}"

                        # 这种路径拼接方式方便后续查找
                        current_save_dir = os.path.join(root_save_dir, mtype, norm_desc, param_desc)
                        os.makedirs(current_save_dir, exist_ok=True)
                        
                        print(f"\n[RUN] 路径: {current_save_dir}")
                        print(f"\n[RUN] 方式: {NORM_NAMES[n_type]} | 模型: {mtype}")
                        
                        all_accs = []
                        p_count = 0
                        for s in SEEDS:
                            acc, p_count = train_one_seed(
                                target_data_path, s, current_save_dir, 
                                mtype, t_down, m_val, h_val, 
                                LR, BATCH_SIZE, EPOCHS, n_type
                            )
                            all_accs.append(acc)
                        
                        avg_acc = np.mean(all_accs)
                        
                        # 构造汇总行
                        row = {
                            "数据源": data_folder_name,
                            "预处理方式": NORM_NAMES[n_type],
                            "模型类型": mtype,
                            "参数量": p_count,
                            "下采样(T)": t_down,
                            "点数(M)": m_val,
                            "Hidden": h_val,
                            "平均准确率": round(avg_acc, 3),
                            "种子结果": str(all_accs)
                        }
                        summary_data.append(row)
                        
                        # 每一组跑完实时保存 Excel
                        # 3. 汇总 Excel 保存在 root_save_dir 下
                        excel_path = os.path.join(root_save_dir, f"Summary_{data_folder_name}.xlsx")
                        pd.DataFrame(summary_data).to_excel(excel_path, index=False)