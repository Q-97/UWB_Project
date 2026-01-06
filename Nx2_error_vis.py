import json
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_errors(json_path="classification_errors.json", num_samples=6):
    """
    读取错误日志并可视化点云（适配新 JSON 格式）
    """
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # --- 关键修改：从嵌套结构中提取 errors 列表 ---
    error_log = data.get('errors', [])
    best_acc = data.get('best_acc', 'N/A')

    if not error_log:
        print(f"没有分类错误的数据可以展示（最高准确率: {best_acc}）。")
        return

    print(f"正在可视化最高准确率轮次 ({best_acc}) 的错误样本...")

    display_count = min(len(error_log), num_samples)
    cols = 3
    rows = (display_count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if display_count > 1 else [axes]

    for i in range(display_count):
        error = error_log[i]
        points = np.array(error['points']) 
        valid_points = points[np.any(points != 0, axis=1)]
        
        ax = axes[i]
        if len(valid_points) > 0:
            ax.scatter(valid_points[:, 0], valid_points[:, 1], alpha=0.6, s=15, c='red')
        
        ax.set_title(f"File: {error['file']}\nTrue: {error['true_label']} | Pred: {error['pred_label']}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
        ax.set_xlim([-2.0, 2.0]) # 建议与训练脚本中的 X_MIN, X_MAX 保持一致
        ax.set_ylim([-2.5, 0.5]) # 建议与训练脚本中的 Y_MIN, Y_MAX 保持一致

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("error_visualization.png")
    print(f"可视化完成。图像已保存为 error_visualization.png")

def save_all_error_plots(json_path="classification_errors.json", output_dir="error_analysis"):
    """
    按文件名分类保存所有错误样本的图像（适配新 JSON 格式）
    """
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # --- 关键修改：从嵌套结构中提取 errors 列表 ---
    error_log = data.get('errors', [])
    best_acc = data.get('best_acc', 'N/A')

    if not error_log:
        print("没有错误数据。")
        return

    print(f"最高准确率: {best_acc}。开始处理 {len(error_log)} 条错误数据...")

    # 1. 按文件来源对错误进行分组
    grouped_errors = {}
    for entry in error_log:
        fname = entry['file']
        if fname not in grouped_errors:
            grouped_errors[fname] = []
        grouped_errors[fname].append(entry)

    # 2. 遍历并绘图
    for fname, samples in grouped_errors.items():
        sub_folder = os.path.join(output_dir, fname.split('.')[0])
        os.makedirs(sub_folder, exist_ok=True)
        
        for idx, error in enumerate(samples):
            points = np.array(error['points'])
            valid_points = points[np.any(points != 0, axis=1)]

            plt.figure(figsize=(6, 6))
            if len(valid_points) > 0:
                plt.scatter(valid_points[:, 0], valid_points[:, 1], alpha=0.6, s=15, c='red')

            plt.title(f"Sample_{idx} | True: {error['true_label']} | Pred: {error['pred_label']}")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.gca().set_aspect('equal')
            plt.xlim([-2.0, 2.0])
            plt.ylim([-2.5, 0.5])

            save_name = f"err_{idx}_T{error['true_label']}_P{error['pred_label']}.png"
            plt.savefig(os.path.join(sub_folder, save_name))
            plt.close()

    print(f"\n[完成] 所有图像已保存至目录: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 根据需要选择执行
    save_all_error_plots()