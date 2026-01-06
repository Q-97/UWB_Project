import json
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_errors(json_path="classification_errors_MxTx2.json", output_root="error_plots"):
    if not os.path.exists(json_path):
        print(f"找不到文件: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    errors = data.get("errors", [])
    if not errors:
        print("没有发现错误样本。")
        return

    os.makedirs(output_root, exist_ok=True)

    for i, err in enumerate(errors):
        # 按文件名创建子目录
        file_name = err["file"].replace(".json", "")
        save_dir = os.path.join(output_root, file_name)
        os.makedirs(save_dir, exist_ok=True)

        # 准备数据 (T, M, 2)
        points = np.array(err["points"]) 
        T, M, _ = points.shape
        
        plt.figure(figsize=(8, 6))
        
        # 遍历每一帧进行绘制，颜色随时间 T 变化
        for t in range(T):
            frame_points = points[t] # (M, 2)
            # 过滤掉填充的零点
            valid_mask = np.any(frame_points != 0, axis=1)
            valid_points = frame_points[valid_mask]
            
            if len(valid_points) > 0:
                # 使用渐变色表示时间流逝 (从浅到深)
                color = plt.cm.Blues(0.3 + 0.7 * (t / T))
                plt.scatter(valid_points[:, 0], valid_points[:, 1], 
                            color=color, s=20, alpha=0.6)

        plt.title(f"Error Sample: {file_name}\nWindow: {err['window_idx']} | True: {err['true_label']} | Pred: {err['pred_label']}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.xlim(-2, 2)
        plt.ylim(-2.5, 0.5)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 保存图片
        plot_name = f"win_{err['window_idx']}_T{err['true_label']}_P{err['pred_label']}.png"
        plt.savefig(os.path.join(save_dir, plot_name))
        plt.close()

    print(f"可视化完成，图片保存在: {output_root}")

if __name__ == "__main__":
    visualize_errors(r"D:\uwb occupy data collection\Res_T81_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\errors_seed2025_acc97.59.json",
                     r"D:\uwb occupy data collection\Res_T81_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\error_plots_2025")
