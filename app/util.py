from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks

def map_tofs_to_space(
    estimated_tofs: List[float], 
    tx_pos: np.ndarray, 
    rx_pos: np.ndarray, 
    grid_x: np.ndarray, 
    grid_y: np.ndarray, 
    c0: float = 2.99792458e8, 
    ring_width_time: float = 0.5e-9
) -> np.ndarray:
    """
    将一个或多个离散的飞行时间(ToF)映射到二维空间，形成椭圆环。
    """
    spatial_fp_image = np.zeros((len(grid_y), len(grid_x)))
    if not estimated_tofs:
        return spatial_fp_image

    variance = ring_width_time**2

    for i, y in enumerate(grid_y):
        for j, x in enumerate(grid_x):
            p = np.array([x, y, 0])
            # 计算该像素对应的理论ToF
            total_dist = np.linalg.norm(p - tx_pos) + np.linalg.norm(p - rx_pos)
            pixel_toa = total_dist / c0
            
            # 找到与该像素ToA最近的已检测ToF
            min_time_diff = np.min(np.abs(np.array(estimated_tofs) - pixel_toa))
            
            # 使用高斯函数为像素赋值。当像素ToA接近检测到的ToF时，值接近1
            pixel_value = np.exp(-min_time_diff**2 / (2 * variance))
            spatial_fp_image[i, j] = pixel_value
            
    return spatial_fp_image


# Add this new function to your fptrm_algorithm.py file
# You'll need these helper functions from your module as well
# Make sure they are available in the same file or imported correctly
def calculate_eigen_decomposition(K):
    TRO = K.conj().T @ K
    eigenvalues, eigenvectors = np.linalg.eigh(TRO)
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

def backproject_tr_music_image(null_space_eigenvectors, imaging_grid, antenna_pos, freq, epsilon_r):
    from scipy.special import hankel1 # Keep import local for multiprocessing
    c0 = 2.99792458e8
    x_coords, y_coords = imaging_grid
    image = np.zeros((len(y_coords), len(x_coords)))
    k = 2 * np.pi * freq / (c0 / np.sqrt(epsilon_r))
    P_N = null_space_eigenvectors @ null_space_eigenvectors.conj().T


    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            pixel_pos = np.array([x, y])
            r = np.linalg.norm(antenna_pos - pixel_pos, axis=1)
            steering_vector = (1j / 4) * hankel1(0, k * r)

            steering_vector /= np.linalg.norm(steering_vector)
            proj_energy = np.abs(np.vdot(steering_vector, P_N @ steering_vector))
            image[i, j] = 1 / (proj_energy + 1e-9)
    return image

def run_trm_imaging(
    mdm_diff_freq, freq_axis_si,
    tx_positions_si, rx_positions_si, freq_range_si, epsilon_r_background,
    num_targets, imaging_grid, num_workers=None, cf = 7.9872e9
):
    """
    单独运行TR-MUSIC算法，生成角度估计图。
    [V3: 计算并打印每个频点的特征值信噪比(SNR)]
    """
    min_freq, max_freq = freq_range_si
    selected_indices = np.where((freq_axis_si >= min_freq) & (freq_axis_si <= max_freq) & (freq_axis_si != 0))[0]
    grid_x, grid_y = imaging_grid
    
    trm_image = np.zeros((len(grid_y), len(grid_x)))

    # --- 修改点: 更新打印内容的标题 ---
    print("\n--- Eigenvalue SNR Analysis per Frequency ---")
    snr_sum = 0
    for freq_idx in selected_indices:
        freq = freq_axis_si[freq_idx]
        if freq == 0: continue
            
        K = mdm_diff_freq[:, :, freq_idx]
        
        # 假设这个函数返回降序排列的特征值和对应的特征向量
        eigenvalues, eigenvectors = calculate_eigen_decomposition(K)
        
        # ========================== 新增/修改的核心逻辑开始 ==========================
        
        # 检查是否有足够的特征值来分离信号和噪声
        if len(eigenvalues) > num_targets:
            # 前 num_targets 个是信号特征值
            signal_eigenvalues = eigenvalues[:num_targets]
            # 剩下的是噪声特征值
            noise_eigenvalues = eigenvalues[num_targets:]

            # 计算信号和噪声的平均功率（这里用总和也可以，比率是相同的）
            total_signal_power = np.sum(signal_eigenvalues)
            total_noise_power = np.sum(noise_eigenvalues[:num_targets])
            
            # 计算信噪比 (SNR)
            # 为避免除以零，检查噪声功率是否接近于零
            if total_noise_power > 1e-9:
                snr = total_signal_power / total_noise_power
                snr_db = 10 * np.log10(snr)
                snr_sum+=snr_db
                #(f"  - Freq: {freq / 1e6:7.2f} MHz, SNR: {snr:8.2f} ({snr_db:5.1f} dB)")
            else:
                # 如果噪声功率为0，信噪比理论上是无穷大
                print(f"  - Freq: {freq / 1e6:7.2f} MHz, SNR: Inf (Noise power is zero)")
        else:
            # 如果特征值数量不足，则无法计算信噪比
            print(f"  - Freq: {freq / 1e6:7.2f} MHz, Not enough eigenvalues to calculate SNR.")

        

        # ========================== 新增/修改的核心逻辑结束 ==========================
        
        if K.shape[1] <= num_targets: continue
        null_space_eigenvectors = eigenvectors[:, num_targets:]
        if null_space_eigenvectors.shape[1] == 0: continue

        single_freq_image = backproject_tr_music_image(
            null_space_eigenvectors,
            imaging_grid,
            rx_positions_si[:, :2], # Use only X,Y coordinates
            freq, #
            epsilon_r_background
        )
        trm_image += single_freq_image
    snr_mean = snr_sum/len(selected_indices)
    print("mean snr:",snr_mean)
    print("--- End of Eigenvalue SNR Analysis ---\n")

    # 归一化最终图像以便更好地可视化
    if np.max(trm_image) > 0:
        trm_image = trm_image
        
    return trm_image,snr_mean


def run_trm_imaging_avg(
    cir_snapshots, # <--- 修改点: 接收 (N_rx, N_tx, N_taps, N_snapshots)
    freq_axis_si,
    tx_positions_si, rx_positions_si, freq_range_si, epsilon_r_background,
    num_targets, imaging_grid, num_workers=None, cf = 7.9872e9
):
    """
    单独运行TR-MUSIC算法...
    [V4: 增加多快照平均功能]
    """
    min_freq, max_freq = freq_range_si
    selected_indices = np.where((freq_axis_si >= min_freq) & (freq_axis_si <= max_freq) & (freq_axis_si != 0))[0]
    grid_x, grid_y = imaging_grid
    
    trm_image = np.zeros((len(grid_y), len(grid_x)))

# --- 修改点: 在循环外先进行一次FFT ---
    # mdm_diff_freq_all 现在的形状是 (N_rx, N_tx, N_taps, N_snapshots)
    mdm_diff_freq_all = np.fft.fft(cir_snapshots, axis=2)
    snr_sum = 0
    for freq_idx in selected_indices:
        freq = freq_axis_si[freq_idx]
        if freq == 0: continue
            
        # --- 核心修改：平均 K 矩阵 ---
        # 1. 提取所有快照在该频率点的数据
        # K_stack 的形状是 (N_rx, N_tx, N_snapshots)
        K_stack = mdm_diff_freq_all[:, :, freq_idx, :]
        
        # 2. 沿着慢时间轴 (axis=2) 对 K 矩阵进行相干平均
        K_avg = np.mean(K_stack, axis=2) 
        # --- 核心修改结束 ---

        # 3. 使用平均后的 K_avg 进行后续计算
        eigenvalues, eigenvectors = calculate_eigen_decomposition(K_avg) # K_avg 替代 K
        
        # ========================== 新增/修改的核心逻辑开始 ==========================
        
        # 检查是否有足够的特征值来分离信号和噪声
        if len(eigenvalues) > num_targets:
            # 前 num_targets 个是信号特征值
            signal_eigenvalues = eigenvalues[:num_targets]
            # 剩下的是噪声特征值
            noise_eigenvalues = eigenvalues[num_targets:]

            # 计算信号和噪声的平均功率（这里用总和也可以，比率是相同的）
            total_signal_power = np.sum(signal_eigenvalues)
            total_noise_power = np.sum(noise_eigenvalues[:num_targets])
            
            # 计算信噪比 (SNR)
            # 为避免除以零，检查噪声功率是否接近于零
            if total_noise_power > 1e-9:
                snr = total_signal_power / total_noise_power
                snr_db = 10 * np.log10(snr)
                snr_sum+=snr_db
                #(f"  - Freq: {freq / 1e6:7.2f} MHz, SNR: {snr:8.2f} ({snr_db:5.1f} dB)")
            else:
                # 如果噪声功率为0，信噪比理论上是无穷大
                print(f"  - Freq: {freq / 1e6:7.2f} MHz, SNR: Inf (Noise power is zero)")
        else:
            # 如果特征值数量不足，则无法计算信噪比
            print(f"  - Freq: {freq / 1e6:7.2f} MHz, Not enough eigenvalues to calculate SNR.")

        

        # ========================== 新增/修改的核心逻辑结束 ==========================
        
        if K_avg.shape[1] <= num_targets: continue
        null_space_eigenvectors = eigenvectors[:, num_targets:]
        if null_space_eigenvectors.shape[1] == 0: continue

        single_freq_image = backproject_tr_music_image(
            null_space_eigenvectors,
            imaging_grid,
            rx_positions_si[:, :2], # Use only X,Y coordinates
            freq, #
            epsilon_r_background
        )
        trm_image += single_freq_image
    snr_mean = snr_sum/len(selected_indices)
    print("mean snr:",snr_mean)
    print("--- End of Eigenvalue SNR Analysis ---\n")

    # 归一化最终图像以便更好地可视化
    if np.max(trm_image) > 0:
        trm_image = trm_image
        
    return trm_image,snr_mean


def run_trm_imaging_r(
    cir_snapshots: np.ndarray,  # <--- [修改] 接收 (N_rx, N_tx, N_taps, N_snapshots)
    freq_axis_si: np.ndarray,
    tx_positions_si: np.ndarray, 
    rx_positions_si: np.ndarray, 
    freq_range_si: tuple, 
    epsilon_r_background: float,
    num_targets: int, 
    imaging_grid: tuple, 
    num_workers: int = None, 
    cf: float = 7.9872e9
):
    """
    [V4 - 非相干平均版]
    单独运行TR-MUSIC算法，生成角度估计图。
    通过平均多个慢时间快照的协方差矩阵 (R = K @ K_H) 来提高稳定性。
    """
    min_freq, max_freq = freq_range_si
    selected_indices = np.where((freq_axis_si >= min_freq) & (freq_axis_si <= max_freq) & (freq_axis_si != 0))[0]
    grid_x, grid_y = imaging_grid
    
    trm_image = np.zeros((len(grid_y), len(grid_x)))

    if cir_snapshots.ndim != 4 or cir_snapshots.shape[3] == 0:
        print("[run_trm_imaging 错误] 输入的 cir_snapshots 维度不正确或快照数为0。")
        return trm_image, 0

    num_snapshots = cir_snapshots.shape[3]
    num_rx = cir_snapshots.shape[0]

    # --- [修改] 在循环外先进行一次FFT ---
    # mdm_diff_freq_all 形状为 (N_rx, N_tx, N_freqs, N_snapshots)
    mdm_diff_freq_all = np.fft.fft(cir_snapshots, axis=2)
    
    # --- 打印标题 ---
    print("\n--- Eigenvalue SNR Analysis (Averaged Covariance) ---")
    snr_sum = 0
    
    for freq_idx in selected_indices:
        freq = freq_axis_si[freq_idx]
        if freq == 0: continue
            
        # --- [核心修改] 平均 R (协方差) 矩阵 ---
        
        # 1. 提取所有快照在该频率点的数据
        # K_stack 形状: (N_rx, N_tx, N_snapshots)
        K_stack = mdm_diff_freq_all[:, :, freq_idx, :]
        
        # 2. 初始化该频率的平均协方差 R (使用 RX 侧, AoA)
        R_avg_f = np.zeros((num_rx, num_rx), dtype=np.complex128)
        
        # 3. 遍历所有快照, 累加协方差
        for l in range(num_snapshots):
            K_l = K_stack[..., l] # (N_rx, N_tx)
            R_avg_f += K_l @ K_l.conj().T # (N_rx, N_rx)
            
        # 4. 求平均
        R_avg_f /= num_snapshots
        # --- 核心修改结束 ---

        # 5. 对平均后的 R_avg_f 进行 EVD (特征值分解)
        try:
            # np.linalg.eigh 返回升序的特征值
            eigenvalues, eigenvectors = np.linalg.eigh(R_avg_f)
        except np.linalg.LinAlgError:
            print(f"  - Freq: {freq / 1e6:7.2f} MHz, EVD 失败，已跳过。")
            continue
            
        # 6. 转换为降序
        eigenvalues = np.abs(eigenvalues)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # ========================== SNR 计算逻辑 (使用 R_avg_f 的特征值) ==========================
        if len(eigenvalues) > num_targets:
            signal_eigenvalues = eigenvalues[:num_targets]
            noise_eigenvalues = eigenvalues[num_targets:]

            # 比较信号功率与等维度的噪声功率
            if len(noise_eigenvalues) >= num_targets:
                total_signal_power = np.sum(signal_eigenvalues)
                total_noise_power = np.sum(noise_eigenvalues[:num_targets])
            
                if total_noise_power > 1e-9:
                    snr = total_signal_power / total_noise_power
                    snr_db = 10 * np.log10(snr)
                    snr_sum += snr_db
                    # (f"  - Freq: {freq / 1e6:7.2f} MHz, SNR: {snr_db:5.1f} dB") # 调试时可打开
                else:
                    # (f"  - Freq: {freq / 1e6:7.2f} MHz, SNR: Inf (Noise power is zero)")
                    pass
            else:
                # (f"  - Freq: {freq / 1e6:7.2f} MHz, Not enough noise eigenvalues.")
                pass
        else:
            # (f"  - Freq: {freq / 1e6:7.2f} MHz, Not enough eigenvalues.")
            pass
        # ========================== SNR 计算结束 ==========================
        
        # 7. 划分噪声子空间 (来自 R_avg_f)
        if R_avg_f.shape[0] <= num_targets: # K.shape[0] (N_rx)
            continue
        
        null_space_eigenvectors = eigenvectors[:, num_targets:]
        if null_space_eigenvectors.shape[1] == 0: 
            continue

        # 8. 空间谱搜索 (使用 RX 天线位置)
        single_freq_image = backproject_tr_music_image(
            null_space_eigenvectors,
            imaging_grid,
            rx_positions_si[:, :2], # <--- 使用 RX 位置 (与 R_avg_f 匹配)
            freq, 
            epsilon_r_background
        )
        trm_image += single_freq_image
        
    snr_mean = snr_sum / len(selected_indices) if len(selected_indices) > 0 else 0
    print(f"--- TR-MUSIC 平均 SNR: {snr_mean:.2f} dB ---")

    # 归一化最终图像以便更好地可视化
    if np.max(trm_image) > 0:
        trm_image = trm_image
        
    return trm_image, snr_mean


# =============================================================================
# [升级] 替换 fptrm_algorithm.py 中的 run_music_angle_scan 函数
# =============================================================================
def run_music_angle_scan(
    mdm_diff_freq, 
    freq_axis_si,
    rx_positions_si,
    freq_range_si,
    num_targets,
    angle_scan_rad,
    snr_threshold   # 新增：信噪比门限
):
    """
    执行一维MUSIC角度扫描，并根据特征值比值判断是否检测到有效目标。

    Returns:
        (numpy.ndarray, bool): (角度谱, 是否检测到目标)
    """
    min_freq, max_freq = freq_range_si
    selected_indices = np.where((freq_axis_si >= min_freq) & (freq_axis_si <= max_freq))[0]
    
    if len(selected_indices) == 0:
        return np.ones_like(angle_scan_rad), False
        
    center_freq_idx = selected_indices[len(selected_indices) // 2]
    freq = freq_axis_si[center_freq_idx]
    
    K = mdm_diff_freq[:, :, center_freq_idx]
    
    if K.shape[0] <= num_targets:
        return np.ones_like(angle_scan_rad), False
        
    R = K @ K.conj().T
    
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    
    # --- 核心修改：在这里进行检测 ---
    # 计算最大特征值与次大特征值的比值
    # 添加一个极小值防止除以零
    eigen_ratio = sorted_eigenvalues[0] / (sorted_eigenvalues[1] + 1e-9) 
    
    is_target_detected = eigen_ratio > snr_threshold
    
    # 如果未检测到目标，直接返回，不进行后续耗时的计算
    if not is_target_detected:
        return np.ones_like(angle_scan_rad), False
    # --------------------------------

    # 如果检测到目标，则继续进行MUSIC计算
    noise_subspace = eigenvectors[:, :len(eigenvectors) - num_targets] #修正噪声子空间选择
    P_N = noise_subspace @ noise_subspace.conj().T

    c0 = 2.99792458e8
    k = 2 * np.pi * freq / c0
    music_spectrum = np.zeros(len(angle_scan_rad))
    rx_pos_xy = rx_positions_si[:, :2]

    for i, theta in enumerate(angle_scan_rad):
        steering_vector = np.exp(-1j * k * (rx_pos_xy[:, 0] * np.sin(theta) + rx_pos_xy[:, 1] * np.cos(theta)))
        proj_energy = np.abs(np.vdot(steering_vector, P_N @ steering_vector))
        music_spectrum[i] = 1 / (proj_energy + 1e-9)
        
    return music_spectrum, True


#找峰值
import numpy as np

def threshold_func(row_idx,start_threshold=1200,default_threshold=400):
    if row_idx == 0:
        return 5000
    elif row_idx == 1:
        return 3500
    else:
        delta = start_threshold - default_threshold
        base = 2/row_idx
        return delta * base + default_threshold


def threshold_func_static(row_idx,start_threshold,default_threshold):
    return default_threshold


def find_peak_indices_np(matrix, threshold_func,start_threshold=1200,default_threshold=400):
    """
    修正版的纯NumPy实现，用于寻找大于阈值的局部最大值（5×5邻域）。
    这个版本修复了边界处理的严重逻辑错误。
    
    参数:
        matrix: 输入的NumPy矩阵
        threshold: 阈值
        
    返回:
        峰值点的索引数组 (N, 2)
    """
    rows, cols = matrix.shape

    # 为每一行创建对应的阈值数组
    thresholds = np.array([threshold_func(i,start_threshold,default_threshold) for i in range(rows)])
    # 将阈值数组扩展为与matrix相同形状
    threshold_matrix = np.tile(thresholds[:, np.newaxis], (1, cols))

    # 找出所有大于阈值的点
    above_threshold = matrix > threshold_matrix
    # 初始化一个掩码，假设所有点都是峰值
    peak_mask = np.ones_like(matrix, dtype=bool)
    
    # 遍历5×cols邻域中的每一个相对位置 (不包括中心点)
    for i in range(-2, 3):
        for j in range(-cols//2, cols//2):
            if i == 0 and j == 0:
                continue
            
            # 创建一个平移后的矩阵副本
            shifted = np.roll(matrix, shift=(i, j), axis=(0, 1))
            
            # --- 这是修正的核心部分 ---
            # 根据平移方向，将“卷回来”的无效数据设置为-inf
            # 这样它们在比较时就不会产生影响
            if i > 0: # 向下平移，顶部i行是无效的
                shifted[:i, :] = -np.inf
            elif i < 0: # 向上平移，底部i行是无效的
                shifted[i:, :] = -np.inf
            
            if j > 0: # 向右平移，左侧j列是无效的
                shifted[:, :j] = -np.inf
            elif j < 0: # 向左平移，右侧j列是无效的
                shifted[:, j:] = -np.inf
            # --------------------------

            # 更新掩码，如果一个点不是大于等于其邻居，就将其标记为False
            # 如果需要严格峰值，请使用 >
            peak_mask &= (matrix >= shifted)
    
    # 最终的峰值必须同时满足是局部最大值且大于阈值
    final_peak_mask = peak_mask & above_threshold
    
    return np.argwhere(final_peak_mask)





# import numpy as np
# from scipy.ndimage import maximum_filter

# def find_peak_indices_np(matrix, threshold):
#     """
#     使用 SciPy 在一个 5行 x N列 的带状邻域内寻找峰值。
#     一个点是峰值，当且仅当它是其所在行及上下各两行范围内的最大值。
    
#     参数:
#         matrix: 输入的NumPy矩阵
#         threshold: 阈值
        
#     返回:
#         峰值点的索引数组 (N, 2)
#     """
#     if matrix.ndim != 2:
#         raise ValueError("输入必须是一个二维矩阵")
    
#     rows, cols = matrix.shape
    
#     # --- 这是实现你想法的核心 ---
#     # 定义邻域的形状为 (5行, 整个矩阵的宽度)
#     footprint_size = (5, cols)
    
#     # 1. 使用最大值滤波器。输出的矩阵中，每个点的值都是其 5xN 带状邻域内的最大值。
#     #    mode='reflect' 是一种常用的边界处理方式，效果通常很好。
#     local_max = maximum_filter(matrix, size=footprint_size, mode='reflect')
    
#     # 2. 如果一个点的值等于其邻域最大值，那么它就是一个局部峰值。
#     peak_mask = (matrix == local_max)
    
#     # 3. 峰值还必须大于指定的阈值。
#     above_threshold = (matrix > threshold)
    
#     # 4. 结合两个条件
#     final_peak_mask = peak_mask & above_threshold
    
#     return np.argwhere(final_peak_mask)


def is_point_in_rectangle(point_x, point_y, rect):
    """
    判断坐标点是否在给定的矩形范围内
    
    参数:
        point_x (float): 测量点的x坐标
        point_y (float): 测量点的y坐标
        rect (plt.Rectangle): matplotlib的矩形对象
    
    返回:
        bool: 如果点在矩形内返回True，否则返回False
    """
    # 获取矩形的左下角坐标和宽高
    rect_x = rect.get_x()
    rect_y = rect.get_y()
    rect_width = rect.get_width()
    rect_height = rect.get_height()
    
    # 判断点是否在矩形范围内
    return (rect_x <= point_x <= rect_x + rect_width and 
            rect_y <= point_y <= rect_y + rect_height)



# --- 2D-MUSIC (MDL) 算法所需的辅助函数 ---

def calculate_mdl_asc(k, eigvals_asc, M_sub, L_prime):
    """
    根据 "V2D-STS-MUSIC" 论文 4.3 节计算 MDL 值。
    假设 eigvals_asc 是 *升序* 排列的特征值。
    k 是假设的信号数量。
    """
    p = M_sub - k  # 噪声子空间的维度
    if p <= 0:
        return np.inf # k >= M_sub，模型无效

    # 噪声特征值是 k 个信号之外的、最小的 p=M_sub-k 个特征值
    noise_eigvals = eigvals_asc[0:p]
    
    # 防止数值问题
    noise_eigvals[noise_eigvals <= 0] = 1e-12 
    
    # 几何平均数
    geom_mean = np.exp(np.mean(np.log(noise_eigvals)))
    # 算术平均数
    arith_mean = np.mean(noise_eigvals)

    if arith_mean < 1e-12:
        return np.inf

    # 对数似然项 (来自文档 4.3)
    log_likelihood = -L_prime * p * np.log(geom_mean / arith_mean)
    
    # 惩罚项 (来自文档 4.3)
    penalty = 0.5 * k * (2 * M_sub - k) * np.log(L_prime)
    
    return log_likelihood + penalty


def find_xy_peaks_from_image(
    image: np.ndarray, 
    k_hat: int, 
    grid_x: np.ndarray, 
    grid_y: np.ndarray,
    min_distance_px: int = 5,
    threshold_rel: float = 0.3
) -> np.ndarray:
    """
    [移植自 simulation_module_2dmusic.py 的 find_peaks_scipy 逻辑]
    从 2D 伪谱图中提取 K_hat 个最强的峰值。
    
    参数:
        image (np.ndarray): 2D 伪谱图。
        k_hat (int): 期望的目标数量 (来自 MDL)。
        grid_x (np.ndarray): X 轴坐标。
        grid_y (np.ndarray): Y 轴坐标。
        min_distance_px (int): 峰值之间的最小像素距离 (用于非极大值抑制)。
        threshold_rel (float): 相对于最大值的相对阈值 (0.0 到 1.0)。
        
    返回:
        np.ndarray: (N, 2) 形状的数组，包含 [x, y] 坐标，N <= k_hat。
    """
    if k_hat == 0:
        return np.array([])
        
    max_val = np.nanmax(image)
    if max_val == 0:
        return np.array([])

    # 1. 应用阈值
    absolute_threshold = max_val * threshold_rel
    threshold_mask = (image >= absolute_threshold)
    
    # 2. 寻找局部最大值 (Non-Maximum Suppression)
    local_max_image = maximum_filter(image, size=min_distance_px)
    local_maxima_mask = (image == local_max_image)
    
    # 3. 结合两者
    peaks_mask = local_maxima_mask & threshold_mask
    candidate_coords_px = np.argwhere(peaks_mask) # (row, col)
    
    if candidate_coords_px.shape[0] == 0:
        return np.array([])

    # 4. 按强度排序并选择 K_hat 个
    intensities = image[candidate_coords_px[:, 0], candidate_coords_px[:, 1]]
    sorted_indices = np.argsort(intensities)[::-1] # 降序
    
    # 取 k_hat 个峰值，或者所有找到的峰值 (如果少于 k_hat)
    num_peaks_to_take = min(k_hat, candidate_coords_px.shape[0])
    final_peak_coords_px = candidate_coords_px[sorted_indices[:num_peaks_to_take]]
    
    # 5. 转换为 (x, y) 坐标
    peak_positions_xy = np.array(
        [[grid_x[c], grid_y[r]] for r, c in final_peak_coords_px]
    )
    
    return peak_positions_xy



def cfar_2d_detector(image: np.ndarray, 
                   training_size: Tuple[int, int], 
                   guard_size: Tuple[int, int], 
                   threshold_factor: float) -> np.ndarray:
    """
    对 2D 图像执行单元平均 (CA-CFAR) 检测。

    参数:
        image (np.ndarray): 输入的 2D 图像 (例如 MUSIC 伪谱)。
        training_size (Tuple[int, int]): (高度, 宽度) - 训练窗口的总大小 (奇数)。
        guard_size (Tuple[int, int]): (高度, 宽度) - 保护窗口的大小 (奇数, 小于 training_size)。
        threshold_factor (float): 阈值乘法因子 (T)。

    返回:
        np.ndarray: 一个布尔掩码 (mask)，'True' 表示检测到的目标。
    """
    
    # 确保窗口大小为奇数
    if training_size[0] % 2 == 0 or training_size[1] % 2 == 0:
        raise ValueError("Training window size 必须为奇数。")
    if guard_size[0] % 2 == 0 or guard_size[1] % 2 == 0:
        raise ValueError("Guard window size 必须为奇数。")

    # 1. 使用 uniform_filter 高效计算“均值”
    # 'constant' 模式表示在图像边缘用 0 填充
    full_mean = uniform_filter(image, size=training_size, mode='constant', cval=0)
    guard_mean = uniform_filter(image, size=guard_size, mode='constant', cval=0)
    
    # 2. 将均值转换回“总和”
    full_sum = full_mean * (training_size[0] * training_size[1])
    guard_sum = guard_mean * (guard_size[0] * guard_size[1])
    
    # 3. 计算训练环带 (training band) 的总和
    training_band_sum = full_sum - guard_sum
    
    # 4. 计算训练环带中的像素数
    num_training_pixels = (training_size[0] * training_size[1]) - (guard_size[0] * guard_size[1])
    
    if num_training_pixels <= 0:
        print("[CFAR 错误] 训练像素数为 0。")
        return np.zeros_like(image, dtype=bool)
        
    # 5. 计算噪声/杂波的平均估计值
    noise_estimate = training_band_sum / num_training_pixels
    
    # 6. 计算每个像素的自适应阈值
    threshold = noise_estimate * threshold_factor
    
    # 7. 将单元格与阈值进行比较
    cfar_mask = image > threshold
    
    return cfar_mask


from typing import List, Tuple, Dict

def create_virtual_array_mapping(
    tx_positions: np.ndarray, 
    rx_positions: np.ndarray,
    tx_indices_to_use: List[int],
    rx_indices_to_use: List[int],
    use_phase_center_convention: bool = True,
    decimal_precision: int = 4
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    计算 MIMO 虚拟阵列映射，自动处理重叠的元素。

    参数:
        tx_positions (np.ndarray): 完整的 TX 天线位置数组 (N_tx_total, 3).
        rx_positions (np.ndarray): 完整的 RX 天线位置数组 (N_rx_total, 3).
        tx_indices_to_use (List[int]): 要使用的 TX 天线的索引列表 (0-based).
        rx_indices_to_use (List[int]): 要使用的 RX 天线的索引列表 (0-based).
        use_phase_center_convention (bool): 
            True - 使用 (P_t + P_r) / 2 作为虚拟位置 (相位中心).
            False - 使用 P_t + P_r 作为虚拟位置.
        decimal_precision (int): 用于检测重叠位置的小数精度。

    返回:
        Tuple[List[Tuple[int, int]], np.ndarray]:
            - selected_pairs: 一个 (tx_idx, rx_idx) 元组的列表，代表唯一的虚拟元素。
            - unique_v_positions: 一个 (N_virtual, 3) 的 NumPy 数组，包含唯一的虚拟元素位置。
    """
    # 字典用于存储 {虚拟位置: (tx_idx, rx_idx)}，自动处理重叠
    virtual_positions_map: Dict[Tuple, Tuple[int, int]] = {}
    
    # 决定是使用相位中心 (P_t+P_r)/2 还是 (P_t+P_r)
    pos_factor = 0.5 if use_phase_center_convention else 1.0
    
    for tx_idx in tx_indices_to_use:
        if tx_idx >= len(tx_positions):
            print(f"[create_virtual_array_mapping] 警告: TX 索引 {tx_idx} 超出范围，已跳过。")
            continue
        for rx_idx in rx_indices_to_use:
            if rx_idx >= len(rx_positions):
                print(f"[create_virtual_array_mapping] 警告: RX 索引 {rx_idx} 超出范围，已跳过。")
                continue
                
            tx_pos = tx_positions[tx_idx]
            rx_pos = rx_positions[rx_idx]
            
            # 计算虚拟位置
            v_pos = (tx_pos + rx_pos) * pos_factor
            
            # 四舍五入以处理浮点不精确性
            v_pos_rounded = tuple(np.round(v_pos, decimal_precision))
            
            if v_pos_rounded not in virtual_positions_map:
                # 这是一个新的、唯一的虚拟元素
                virtual_positions_map[v_pos_rounded] = (tx_idx, rx_idx)
    
    # 根据虚拟位置对(key)进行排序 (例如，按 x 坐标)
    # item[0] 是虚拟位置元组, item[0][0] 是 x 坐标
    sorted_items = sorted(virtual_positions_map.items(), key=lambda item: item[0][0])
    
    # 提取排序后的 (tx, rx) 对和虚拟位置
    selected_pairs = [item[1] for item in sorted_items]
    unique_v_positions = np.array([list(item[0]) for item in sorted_items])
    
    return selected_pairs, unique_v_positions


def find_first_peak(matrix):
    """
    寻找第一个维度（行）最小的峰值（5×5邻域内的局部最大值）
    
    参数:
        matrix: 输入的NumPy矩阵
        
    返回:
        峰值点的索引数组 (1, 2)，如果没有找到则返回空数组
    """
    rows, cols = matrix.shape

    # 初始化峰值掩码
    peak_mask = np.ones_like(matrix, dtype=bool)
    
    # 遍历5×5邻域中的每一个相对位置 (不包括中心点)
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i == 0 and j == 0:
                continue
            
            # 创建一个平移后的矩阵副本
            shifted = np.roll(matrix, shift=(i, j), axis=(0, 1))
            
            # 边界处理：将"卷回来"的无效数据设置为负无穷
            if i > 0:  # 向下平移，顶部i行无效
                shifted[:i, :] = -np.inf
            elif i < 0:  # 向上平移，底部i行无效
                shifted[i:, :] = -np.inf
            
            if j > 0:  # 向右平移，左侧j列无效
                shifted[:, :j] = -np.inf
            elif j < 0:  # 向左平移，右侧j列无效
                shifted[:, j:] = -np.inf

            # 更新掩码：只有大于等于所有邻居的点才可能是峰值
            peak_mask &= (matrix >= shifted)
    
    # 获取所有峰值点的索引
    peak_indices = np.argwhere(peak_mask)
    
    # 如果没有找到峰值，返回空数组
    if len(peak_indices) == 0:
        return np.array([]).reshape(0, 2)
    
    # 按行号排序，找到第一个维度（行）最小的峰值
    min_row = np.min(peak_indices[:, 0])
    min_row_peaks = peak_indices[peak_indices[:, 0] == min_row]
    
    # 在最小行中选择列号最小的峰值
    first_peak = min_row_peaks[np.argmin(min_row_peaks[:, 1])]
    
    # 返回形状为(1, 2)的数组
    return first_peak.reshape(1, 2)


def find_cir_peaks_simple(cir_signal, min_height_ratio=0.3):
    """
    简化的CIR峰值检测函数
    
    参数:
        cir_signal: CIR波形信号 (1D数组)
        min_height_ratio: 相对于最大值的峰值最小高度比例 (0-1)
        
    返回:
        peak_indices: 峰值索引数组
        peak_values: 峰值幅度数组
    """
    # 计算最小高度阈值
    max_val = np.max(cir_signal)
    min_height = max_val * min_height_ratio
    
    # 寻找峰值
    peaks, properties = find_peaks(cir_signal, height=min_height, distance=5)
    
    # 获取峰值对应的值
    peak_values = cir_signal[peaks]
    
    return peaks, peak_values

def analyze_cir_peaks(cir_signal, min_height_ratio=0.1):
    """
    分析CIR波形的峰值信息
    
    参数:
        cir_signal: CIR波形信号
        min_height_ratio: 最小高度比例阈值
        
    返回:
        dict: 包含峰值详细信息的字典
    """
    # 寻找峰值
    peak_indices, peak_values = find_cir_peaks_simple(cir_signal, min_height_ratio)
    
    if len(peak_indices) == 0:
        return {
            'peak_indices': np.array([]),
            'peak_values': np.array([]),
            'main_peak_idx': None,
            'main_peak_value': None,
            'num_peaks': 0
        }
    
    # 找到主峰值（最大值）
    main_peak_idx = peak_indices[np.argmax(peak_values)]
    main_peak_value = np.max(peak_values)
    
    return {
        'peak_indices': peak_indices,
        'peak_values': peak_values,
        'main_peak_idx': main_peak_idx,
        'main_peak_value': main_peak_value,
        'num_peaks': len(peak_indices)
    }


def generate_gaussian_blobs(coords, grid_x, grid_y, sigma=0.15):
    """
    辅助函数：在指定的 (x, y) 坐标处生成高斯光斑，用于模拟类似 MUSIC 的伪谱输出。
    """
    image = np.zeros((len(grid_y), len(grid_x)))
    if not coords:
        return image
    
    # 创建网格
    X, Y = np.meshgrid(grid_x, grid_y)
    
    for (px, py) in coords:
        # 高斯分布公式
        blob = np.exp(-((X - px)**2 + (Y - py)**2) / (2 * sigma**2))
        image += blob
        
    # 归一化
    # max_val = np.max(image)
    # if max_val > 0:
    #     image /= max_val
    return image
