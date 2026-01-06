import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

def visualize_errors_animated(json_path="classification_errors_MxTx2.json", 
                            output_root="error_animations", 
                            history_len=5): # 新增参数：控制残留帧数
    if not os.path.exists(json_path):
        print(f"找不到文件: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    errors = data.get("errors", [])
    if not errors:
        print("没有发现错误样本。")
        return

    os.makedirs(output_root, exist_ok=True)

    for i, err in enumerate(errors):
        file_name = err["file"].replace(".json", "")
        save_dir = os.path.join(output_root, file_name)
        os.makedirs(save_dir, exist_ok=True)

        # 准备数据 (T, M, 2)
        points = np.array(err["points"]) 
        T, M, _ = points.shape
        
        fig, ax = plt.subplots(figsize=(8, 6))

        # 初始化绘图对象
        # 注意：这里不再设置固定的颜色，因为颜色/透明度会随历史跨度改变
        scat = ax.scatter([], [], s=40)

        def init():
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2.5, 0.5)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, linestyle='--', alpha=0.5)
            return scat,

        def update(t):
            # 确定历史窗口范围 [start_t, t]
            start_t = max(0, t - history_len + 1)
            
            all_points = []
            all_colors = []

            # 遍历窗口内的每一帧，收集点并计算对应的透明度
            for hist_idx in range(start_t, t + 1):
                frame_points = points[hist_idx]
                valid_mask = np.any(frame_points != 0, axis=1)
                valid_points = frame_points[valid_mask]

                if len(valid_points) > 0:
                    all_points.append(valid_points)
                    
                    # 计算权重 (0 到 1 之间)，当前时刻 t 的权重为 1.0
                    # 历史越久，权重越低
                    if t == start_t:
                        weight = 1.0
                    else:
                        weight = (hist_idx - start_t) / (t - start_t)
                    
                    # 线性映射透明度：最新帧 alpha=0.9，最旧帧 alpha=0.1
                    alpha = 0.1 + 0.8 * weight
                    # 颜色：使用 Blues 映射，当前帧颜色深
                    base_color = plt.cm.Blues(0.4 + 0.6 * (hist_idx / T))
                    
                    # 构造 RGBA 颜色，并重复对应点的次数
                    rgba = list(base_color)
                    rgba[3] = alpha # 修改透明度通道
                    all_colors.extend([rgba] * len(valid_points))

            if all_points:
                # 合并所有帧的点
                combined_points = np.vstack(all_points)
                scat.set_offsets(combined_points)
                scat.set_facecolors(all_colors)
            else:
                # 若无有效点，清空显示
                scat.set_offsets(np.empty((0, 2)))

            ax.set_title(f"Error Sample: {file_name}\n"
                         f"Window: {err['window_idx']} | True: {err['true_label']} | Pred: {err['pred_label']}\n"
                         f"Frame: {t}/{T} (Residual: {history_len})")
            return scat,

        # 创建动画
        ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=100)

        # 保存动画
        anim_name = f"win_{err['window_idx']}_T{err['true_label']}_P{err['pred_label']}_res{history_len}.gif"
        save_path = os.path.join(save_dir, anim_name)
        
        try:
            writer = PillowWriter(fps=10)
            ani.save(save_path, writer=writer)
            print(f"已生成动画: {save_path}")
        except Exception as e:
            print(f"保存动画失败: {e}")
        finally:
            plt.close(fig)

    print(f"\n所有可视化完成，动画保存在: {output_root}")

if __name__ == "__main__":
    # 1. 定义 JSON 文件列表（修正了漏掉的逗号）
    json_file_list = [
        r"D:\uwb occupy data collection\Res_T80_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\errors_seed42_acc97.33.json",
        r"D:\uwb occupy data collection\Res_T80_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\errors_seed123_acc97.11.json",
        r"D:\uwb occupy data collection\Res_T80_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\errors_seed2025_acc97.03.json", # 这里加了逗号
        r"D:\uwb occupy data collection\Res_T81_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\errors_seed42_acc97.25.json",
        r"D:\uwb occupy data collection\Res_T81_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\errors_seed123_acc97.51.json",
        r"D:\uwb occupy data collection\Res_T81_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\errors_seed2025_acc97.59.json"
    ]

    # 2. 定义对应的输出目录列表
    output_dir_list = [
        r"D:\uwb occupy data collection\Res_T80_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\error_animations_42",
        r"D:\uwb occupy data collection\Res_T80_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\error_animations_123",
        r"D:\uwb occupy data collection\Res_T80_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\error_animations_2025",
        r"D:\uwb occupy data collection\Res_T81_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\error_animations_42",
        r"D:\uwb occupy data collection\Res_T81_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\error_animations_123",
        r"D:\uwb occupy data collection\Res_T81_M4_B32_LR0.0003333333333333333_E40_X-2_2_Y-2.5_0\error_animations_2025"
    ]

    # 3. 使用 zip 并行遍历，实现一一对应输入
    # zip 会成对取出 (json_file_list[0], output_dir_list[0]), 然后是 [1], 以此类推
    for json_path, out_path in zip(json_file_list, output_dir_list):
        print(f"\n正在处理文件: {os.path.basename(json_path)}")
        print(f"输出目标目录: {out_path}")
        
        # 调用之前定义的函数
        visualize_errors_animated(json_path, out_path,history_len=10)

    print("\n--- 所有任务处理完毕 ---")