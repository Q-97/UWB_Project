import json
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 配置参数 (修改此处会自动改变文件夹名称) ---
T_WINDOW = 64
N_POINTS = 128
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX = -2.5, 0

# --- 2. 动态生成挂钩参数的文件夹名称 ---
# 生成格式如: visualized_samples_T64_N128_X-2to2_Y-2.5to0
OUTPUT_ROOT = f"visualized_samples_T{T_WINDOW}_N{N_POINTS}_X{X_MIN}to{X_MAX}_Y{Y_MIN}to{Y_MAX}"

# --- 抽样配置 ---
SAMPLE_RATE = 0.1  # 抽样比例：10%
RANDOM_SEED = 42

def process_and_visualize_dataset(data_dir, output_dir):
    """
    遍历数据集，按比例抽样并可视化点云
    """
    random.seed(RANDOM_SEED)
    
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print(f"错误: 路径 '{data_dir}' 下没有找到 JSON 文件。")
        return

    print(f"开始处理数据集，输出目录: {output_dir}")
    total_visualized = 0

    for fp in json_files:
        filename = os.path.basename(fp)
        file_tag = filename.split('.')[0]
        
        # 加载数据
        with open(fp, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        all_frames = content.get('data', [])
        if len(all_frames) < T_WINDOW:
            continue
            
        # 滑动窗口预处理
        valid_windows = []
        for i in range(len(all_frames) - T_WINDOW + 1):
            window_frames = all_frames[i : i + T_WINDOW]
            combined_points = []
            for frame in window_frames:
                raw_points = frame.get('points', [])
                # 坐标过滤（严格匹配训练脚本参数）
                filtered = [
                    p for p in raw_points 
                    if X_MIN <= p['pos'][0] <= X_MAX and Y_MIN <= p['pos'][1] <= Y_MAX
                ]
                combined_points.extend(filtered)
            
            # 排序取前 N
            combined_points.sort(key=lambda x: x.get('snr', 0), reverse=True)
            top_points = combined_points[:N_POINTS]
            
            if len(top_points) > 0:
                feature_vec = np.zeros((N_POINTS, 2), dtype=np.float32)
                for j, p in enumerate(top_points):
                    feature_vec[j] = p['pos']
                valid_windows.append((i, feature_vec))

        # 抽样
        num_to_sample = max(1, int(len(valid_windows) * SAMPLE_RATE))
        sampled_data = random.sample(valid_windows, num_to_sample)
        
        # 按照文件类别分类存储
        sub_folder = os.path.join(output_dir, file_tag)
        os.makedirs(sub_folder, exist_ok=True)

        for idx, (win_idx, points) in enumerate(sampled_data):
            # 过滤全零点以便观察
            valid_mask = np.any(points != 0, axis=1)
            pts_to_plot = points[valid_mask]

            plt.figure(figsize=(6, 6))
            if len(pts_to_plot) > 0:
                plt.scatter(pts_to_plot[:, 0], pts_to_plot[:, 1], 
                            alpha=0.6, s=15, c='blue')

            plt.title(f"{file_tag} | Win_{win_idx}")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.gca().set_aspect('equal')
            plt.xlim([X_MIN - 0.2, X_MAX + 0.2])
            plt.ylim([Y_MIN - 0.2, Y_MAX + 0.2])

            plt.savefig(os.path.join(sub_folder, f"sample_{win_idx}.png"), bbox_inches='tight')
            plt.close()
            total_visualized += 1

    print(f"\n[完成] 总计保存 {total_visualized} 张图像至: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 请修改为您的实际数据存放路径
    data_path = r"D:\uwb occupy data collection\数据1229"
    
    if os.path.isdir(data_path):
        process_and_visualize_dataset(data_path, OUTPUT_ROOT)
    else:
        print(f"路径无效: {data_path}")