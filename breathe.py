import numpy as np


def find_breathing_feature(
        doppler_result: np.ndarray,
        cir_fps: float,
        start_freq: float,
        end_freq: float,
        start_path: int,
        stop_path: int
)->tuple[float, int, float]:
    """
        用 Python 复现 C 代码 calc_bfeature_wind 的核心逻辑。

    该函数处理一个复数的Doppler FFT结果矩阵，用于寻找在指定频率范围和路径（tap）
    范围内的最显著的呼吸特征。

    参数:
        doppler_result (np.ndarray): 
            输入的二维复数数组，由 np.fft.fft 计算得出。
            形状为 (num_taps, fft_len)，即 (路径数, 快照数)。
        cir_fps (float): 
            CIR的采样率 (snapshots per second), 相当于 C 代码中的 CIR_FPS。
        start_freq (float): 
            要分析的起始频率 (Hz)，相当于 C 代码中的 START_FREQ。
        end_freq (float): 
            要分析的结束频率 (Hz)，相当于 C 代码中的 END_FREQ。
        start_path (int): 
            开始处理的路径索引 (tap index)。
        stop_path (int): 
            结束处理的路径索引 (不包含此索引)。

    返回:
        tuple[float, int, float]: 包含三个值的元组
        - v_selected (float): 在指定频段内找到的最大归一化能量值。
        - path_selected (int): v_selected 所在的路径(tap)索引。
        - freq_selected (float): v_selected 所在的频率(Hz)。
    """
    # 从输入数据中获取维度信息
    num_taps, fft_len = doppler_result.shape
    # 初始化 C 代码中的变量
    v_selected = 0.0
    path_selected = -1
    freq_selected = 0.0
    first_path_selected = -1
    # 1. 将频率(Hz)转换为 FFT 的索引
    # C 代码: FREQ_TO_IDX(F, CIR_FPS, N) (N*F/CIR_FPS)
    # 我们只考虑正频率部分，FFT结果是对称的
    start_idx = int(fft_len * start_freq / cir_fps)
    end_idx = int(fft_len * end_freq / cir_fps)

    # 确保索引在有效范围内 (0 到 fft_len/2)
    if start_idx < 1: start_idx = 1 # C代码不计算直流分量，所以从1开始
    if end_idx > fft_len // 2: end_idx = fft_len // 2

    # 确保索引在有效范围内 (0 到 fft_len/2)
    if start_idx < 1: start_idx = 1 # C代码不计算直流分量，所以从1开始
    if end_idx > fft_len // 2: end_idx = fft_len // 2
        
    #print(f"频率范围 [{start_freq}, {end_freq}] Hz 对应的FFT索引范围: [{start_idx}, {end_idx})")
    max_motion = 0
    # 2. 遍历指定的每一个 path (tap)
    for i in range(start_path, stop_path):
        
        # 首先，从复数FFT结果中计算幅度谱
        # 这对应 C 代码中的 fft_user_amp2amp_dsp 步骤
        amplitude_spectrum = np.abs(doppler_result[i, :])

        # 3. 计算每个 tap 的正频率能量范数 (norm) 和归一化因子 E
        # C 代码: float norm = calc_norm(cir_amp_st_fft[i]+1, fft_len/2-1);
        # 我们只考虑正频率部分(不包括直流分量 k=0 和奈奎斯特频率 k=fft_len/2)
        # 即索引范围是 [1, fft_len/2 - 1]
        positive_freq_amps = amplitude_spectrum[0 : fft_len//2]
        
        # calc_norm 的过程是计算L2范数：sqrt(sum(x^2))
        norm = np.linalg.norm(positive_freq_amps)
        if max_motion < norm:
            max_motion = norm
        
        # C 代码: E[i] = norm*norm/fft_len + 0.000001;
        E_i = (norm**2 / fft_len) + 1e-6
    

            # 4. 遍历指定频率范围的索引
        for j in range(start_idx, end_idx):
            
            # 5. 计算归一化后的能量值
            # C 代码: cir_amp_st_fft[i][j]*cir_amp_st_fft[i][j] / (fft_len*E[i]);
            # amplitude_spectrum[j]**2 是该频率点的能量
            normalized_energy = (amplitude_spectrum[j]**2) / (fft_len * E_i)
            
            # 6. 判断是否为最大值，并更新结果
            if v_selected < normalized_energy:
                v_selected = normalized_energy
                if first_path_selected == -1 and v_selected > 0.35:
                    first_path_selected = i
                path_selected = i

                # C 代码: freq_tmp = IDX_TO_FREQ(j, CIR_FPS, fft_len);
                freq_selected = j * cir_fps / fft_len

    return v_selected, first_path_selected, freq_selected,max_motion


