import json
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 配置参数 (需与 torch_MxTx2.py 保持一致) ---
T_WINDOW = 16
M_POINTS = 4
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX = -2.5, 0

# --- 2. 动态生成挂钩参数的文件夹名称 ---
OUTPUT_ROOT = f"vis_MxT_T{T_WINDOW}_M{M_POINTS}_X{X_MIN}to{X_MAX}_Y{Y_MIN}to{Y_MAX}"

# --- 抽样配置 ---
SAMPLE_RATE = 0.1  # 抽样比例：10%
RANDOM_SEED = 42

def process_and_visualize_MxT(data_dir, output_dir):
    """
    针对 MxTx2 特征格式进行抽样可视化
    """
    random.seed(RANDOM_SEED)
    
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print(f"错误: 路径 '{data_dir}' 下没有找到 JSON 文件。")
        return

    print(f"开始处理数据集 (MxT 模式)，输出目录: {output_dir}")
    total_visualized = 0

    for fp in json_files:
        filename = os.path.basename(fp)
        file_tag = filename.replace(".json", "")
        
        with open(fp, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        all_frames = content.get('data', [])
        if len(all_frames) < T_WINDOW:
            continue
            
        # 3. 滑动窗口预处理 (按照 MxTx2 逻辑提取特征)
        valid_windows = []
        for i in range(len(all_frames) - T_WINDOW + 1):
            window_frames = all_frames[i : i + T_WINDOW]
            
            # 构造 (T, M, 2) 的特征块
            window_feature = np.zeros((T_WINDOW, M_POINTS, 2), dtype=np.float32)
            for t_idx, frame in enumerate(window_frames):
                raw_points = frame.get('points', [])
                # 过滤坐标
                filtered = [p for p in raw_points if X_MIN <= p['pos'][0] <= X_MAX and Y_MIN <= p['pos'][1] <= Y_MAX]
                # 按 SNR 排序
                filtered.sort(key=lambda x: x.get('snr', 0), reverse=True)
                # 取前 M 个点
                for m_idx, p in enumerate(filtered[:M_POINTS]):
                    window_feature[t_idx, m_idx] = p['pos']
            
            valid_windows.append((i, window_feature))

        # 4. 执行抽样
        num_to_sample = max(1, int(len(valid_windows) * SAMPLE_RATE))
        sampled_data = random.sample(valid_windows, num_to_sample)
        
        # 5. 创建分类目录
        sub_folder = os.path.join(output_dir, file_tag)
        os.makedirs(sub_folder, exist_ok=True)

        # 6. 绘图 (带时间渐变)
        for idx, (win_idx, feature_block) in enumerate(sampled_data):
            plt.figure(figsize=(8, 6))
            
            # 遍历时间步 T，颜色随 t 变化
            for t in range(T_WINDOW):
                frame_pts = feature_block[t] # 形状 (M, 2)
                # 过滤全零点
                mask = np.any(frame_pts != 0, axis=1)
                pts_to_plot = frame_pts[mask]
                
                if len(pts_to_plot) > 0:
                    # 使用 Blues 映射，t 越大（时间越晚）颜色越深
                    color = plt.cm.Blues(0.3 + 0.7 * (t / T_WINDOW))
                    plt.scatter(pts_to_plot[:, 0], pts_to_plot[:, 1], 
                                color=color, s=20, alpha=0.5)

            plt.title(f"{file_tag} | Window_{win_idx}\n(Time Flow: Light -> Dark Blue)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.gca().set_aspect('equal')
            plt.xlim([X_MIN - 0.2, X_MAX + 0.2])
            plt.ylim([Y_MIN - 0.2, Y_MAX + 0.2])

            plt.savefig(os.path.join(sub_folder, f"sample_win{win_idx}.png"), bbox_inches='tight')
            plt.close()
            total_visualized += 1

    print(f"\n[完成] 渲染完毕！")
    print(f"总计保存 {total_visualized} 张时序点云图至: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 请修改为您的实际数据存放路径
    data_path = r"D:\uwb occupy data collection\数据1230 db8"
    
    if os.path.isdir(data_path):
        process_and_visualize_MxT(data_path, OUTPUT_ROOT)
    else:
        print(f"路径无效: {data_path}")