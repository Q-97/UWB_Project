"""算法层：AlgorithmProcessor（9 条流水线）+ 算法辅助函数。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import RegularGridInterpolator
from scipy.signal.windows import chebwin

from app import util
from app.breathe import find_breathing_feature
from app.config import RadarConfig
from app.data_processing import RadarDataManager

def apply_range_bin_selection(current_cube, params, return_start_bin=False):
    keep_range = params.get('range_bin_keep_range')
    if keep_range is not None:
        if isinstance(keep_range, str):
            keep_range = keep_range.strip().strip('()[]').replace('\uff0c', ',').split(',')
        if len(keep_range) != 2:
            raise ValueError("range_bin_keep_range must be [start, end]")
        start_bin = int(keep_range[0])
        end_bin = int(keep_range[1])
        max_bin = current_cube.shape[2] - 1
        start_bin = max(0, min(start_bin, max_bin))
        end_bin = max(start_bin, min(end_bin, max_bin))
        selected = current_cube[:, :, start_bin:end_bin + 1, :]
        return (selected, start_bin) if return_start_bin else selected

    drop_front = int(params.get('range_bin_drop_front', 0) or 0)
    if drop_front > 0:
        drop_front = min(drop_front, max(0, current_cube.shape[2] - 1))
        selected = current_cube[:, :, drop_front:, :]
        return (selected, drop_front) if return_start_bin else selected

    leakage_offset = int(params.get('leakage_offset', 0))
    selected = np.roll(current_cube, -leakage_offset, axis=2)
    return (selected, leakage_offset) if return_start_bin else selected

# ==============================================================================
# 4. Algorithm Processor (Logic Ported + Vectorized)
# ==============================================================================
def calculate_mdl_asc_local(k, eigvals_asc, M, L):
    """ 本地 MDL 计算函数，防止 util.py 缺失 """
    p = M - k
    if p <= 0: return np.inf
    noise_evals = eigvals_asc[:p]
    noise_evals = np.maximum(noise_evals, 1e-12) # 避免 log(0)
    geom_mean = np.exp(np.mean(np.log(noise_evals)))
    arith_mean = np.mean(noise_evals)
    if arith_mean < 1e-12: return np.inf
    term1 = -L * p * np.log(geom_mean / arith_mean)
    term2 = 0.5 * k * (2 * M - k) * np.log(L)
    return term1 + term2

class AlgorithmProcessor:
    def __init__(self, config: RadarConfig, data_manager: RadarDataManager):
        self.config = config
        self.dm = data_manager
        self.init_done = False
        self.virtual_pairs = None
        self._reset_algo_state()

    def _reset_algo_state(self):
        self.breathing_window = deque(maxlen=20)
        self.peak_window = deque(maxlen=10)
        self._dbf_cache_fp = None
        self._reset_dubhe_state()

    def _reset_dubhe_state(self):
        self.dubhe_accum = np.zeros((4, 2, 32), dtype=np.complex64)
        self.dubhe_accum_cnt = 0
        self.dubhe_ring = deque(maxlen=50)
        self.dubhe_new_comb_cnt = 0
        self._dbf_sv = None
        self._dbf_angles = None
        self._capon_sv = None
        self._capon_angles = None

    def _init_music_if_needed(self):
        if self.init_done: return
        params = self.config.algo_params['2D-MUSIC']
        tx_indices = params.get('virt_tx_idx', [0, 1])
        rx_indices = params.get('virt_rx_idx', [2, 3])
        try:
            self.virtual_pairs, _ = util.create_virtual_array_mapping(
                self.config.TX_POS_NP, self.config.RX_POS_NP,
                tx_indices, rx_indices, use_phase_center_convention=False
            )
            print(f"[MUSIC] Init: {len(self.virtual_pairs)} virtual elements.")
            self.init_done = True
            self._reset_algo_state()
        except Exception as e:
            print(f"[MUSIC] Init Error: {e}")

    def _run_2d_music_mdl_internal(self, cir_snapshots, selected_pairs, freq_axis, imaging_grid, 
                                   max_targets, fov_degrees, 
                                   epsilon_r=1.0, system_delay=0.0, 
                                   is_forward_backward=False, diag_load=1e-5,cir_offset = 6):
        """
        [完全移植版] 2D-MUSIC 算法 (向量化加速 + 功能全对齐)
        """
        if not cir_snapshots or len(cir_snapshots) == 0:
            return np.zeros((len(imaging_grid[1]), len(imaging_grid[0]))), 0
        if not selected_pairs:
            return np.zeros((len(imaging_grid[1]), len(imaging_grid[0]))), 0

        try:
            L = len(cir_snapshots)
            selected_freqs = freq_axis
            n_freqs = len(selected_freqs)
            
            if n_freqs == 0:
                return np.zeros((len(imaging_grid[1]), len(imaging_grid[0]))), 0
            
            # --- 1. 准备数据矩阵 X (Features x Snapshots) ---
            X_vecs = []
            for l in range(L):
                cir_l = cir_snapshots[l]
                mdm_freq = np.fft.fft(cir_l, axis=2) # (N_rx, N_tx, FT_LEN)
                
                # 提取特定频率的数据 (N_rx, N_tx, n_freqs)
                # 注意：传入的 cir_snapshots 已经是切片过的还是全频段？
                # 这里假设传入的 freq_axis 是已经筛选过的频率点，但 mdm_freq 是全频段
                # 我们需要找到 freq_axis 在 mdm_freq 中的索引
                # 为了简化性能，这里假设外部已经做好了频率映射，或者我们直接取所有点
                # 如果 freq_axis 是物理频率，我们需要重新计算索引。
                # **更高效的做法**：外部计算 FFT 后只传入感兴趣的 bin。
                # 但为了兼容，我们这里假设 freq_axis 对应的就是 cir_l FFT 后的某些 bin。
                
                # 由于结构限制，这里我们简化：重新计算 FFT 并提取。
                # 假设 freq_axis 是物理频率值。
                all_freqs = np.fft.fftfreq(self.config.ft_len-cir_offset, d=self.config.tap_interval_si) + self.config.algo_params["2D-MUSIC"]["center_freq"]
                # 找到最接近的 indices
                indices = []
                for f in selected_freqs:
                    idx = np.argmin(np.abs(all_freqs - f))
                    indices.append(idx)
                
                data_cube_selected = mdm_freq[:, :, indices]
                
                col_vec = []
                for f_i in range(n_freqs):
                    for (tx_idx, rx_idx) in selected_pairs:
                        col_vec.append(data_cube_selected[rx_idx, tx_idx, f_i])
                X_vecs.append(col_vec)
            
            X = np.array(X_vecs).T # (Features, L)
            
            # --- 2. 计算协方差矩阵 R ---
            R = (X @ X.conj().T) / L
            
            # [移植] 前后向平滑
            if is_forward_backward:
                J = np.flip(np.eye(R.shape[0]), axis=0)
                R = 0.5 * (R + J @ R.conj() @ J)

            # [移植] 对角加载
            R += np.eye(R.shape[0]) * diag_load * np.abs(np.trace(R))
            
            # --- 3. 特征分解 & MDL ---
            evals, evecs = np.linalg.eigh(R)
            evals_asc = np.abs(evals) # 升序
            
            # MDL 计算
            mdl_values = []
            M_sub = R.shape[0]
            search_max = min(max_targets + 1, M_sub)
            for k in range(search_max):
                mdl_values.append(calculate_mdl_asc_local(k, evals_asc, M_sub, L))
            
            K_hat = np.argmin(mdl_values) if mdl_values else 0
            
            if (M_sub - K_hat) <= 0:
                return np.zeros((len(imaging_grid[1]), len(imaging_grid[0]))), K_hat

            # 噪声子空间 (最小特征值对应的向量)
            noise_sub = evecs[:, :M_sub - K_hat]
            Pn = noise_sub @ noise_sub.conj().T
            
            # --- 4. 网格搜索 (向量化) ---
            grid_x, grid_y = imaging_grid
            GV_X, GV_Y = np.meshgrid(grid_x, grid_y)
            flat_x = GV_X.ravel()
            flat_y = GV_Y.ravel()
            
            # [移植] FoV 限制
            max_angle = np.deg2rad(fov_degrees / 2.0)
            angles = np.abs(np.arctan2(flat_x, -flat_y))
            valid_mask = (angles <= max_angle) & (flat_y < 0)
            
            spectrum_flat = np.zeros(len(flat_x))
            
            if np.any(valid_mask):
                val_x = flat_x[valid_mask]
                val_y = flat_y[valid_mask]
                
                pixel_pos = np.stack((val_x, val_y, np.zeros_like(val_x)), axis=1)
                v_prop = 2.99792458e8 / np.sqrt(epsilon_r)
                
                # 距离计算 (包含系统延时补偿)
                dists = np.zeros((len(selected_pairs), len(val_x)))
                tx_pos_all = self.config.TX_POS_NP
                rx_pos_all = self.config.RX_POS_NP
                
                for i, (tx_i, rx_i) in enumerate(selected_pairs):
                    d_tx = np.linalg.norm(pixel_pos - tx_pos_all[tx_i], axis=1)
                    d_rx = np.linalg.norm(pixel_pos - rx_pos_all[rx_i], axis=1)
                    # 距离转时间 + 延时补偿
                    dists[i, :] = (d_tx + d_rx) / v_prop + system_delay
                
                # 导向矢量构建
                A_list = []
                for f in selected_freqs:
                    # phase = exp(-j * 2pi * f * time)
                    phase = np.exp(-1j * 2 * np.pi * f * dists)
                    A_list.append(phase)
                
                A_total = np.vstack(A_list)
                
                # MUSIC 谱
                temp = Pn @ A_total
                denom = np.sum(A_total.conj() * temp, axis=0)
                spectrum_flat[valid_mask] = 1.0 / (np.abs(denom) + 1e-12)
                
            return spectrum_flat.reshape(GV_X.shape), K_hat

        except Exception as e:
            print(f"!!! MUSIC 算法内部崩溃: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((len(imaging_grid[1]), len(imaging_grid[0]))), 0

    def step_2d_music(self):
        # 1. 准备
        self._init_music_if_needed()
        all_c = self.dm.get_all_snapshot_as_array('complex')
        all_a = self.dm.get_all_snapshot_as_array('abs')
        if all_c is None: return None

        # 2. 获取参数
        params = self.config.algo_params['2D-MUSIC']
        n_snaps = params['music_snapshots']
        if all_c.shape[3] < n_snaps: return None
        offset = int(params.get('cir_offset', 0))

        all_c  =  all_c[:,:,offset:,:]
        # 提取快照
        cir_snaps = [all_c[..., i] for i in range(-n_snaps, 0)]
        
        # --- [关键修改] 频率计算逻辑 ---
        cf = params['center_freq']
        bw = params['bandwidth']
        min_freq = cf - (bw / 2)
        max_freq = cf + (bw / 2)
        
        # 生成完整的物理频率轴
        full_freq_axis = np.fft.fftfreq(self.config.ft_len-offset, d=self.config.tap_interval_si) + cf
        
        # 筛选出有效频率范围内的频率点
        valid_indices = np.where((full_freq_axis >= min_freq) & (full_freq_axis <= max_freq))[0]
        selected_freq_axis = full_freq_axis[valid_indices]
        
        # 3. 运行核心算法
        trm_image, k_hat = self._run_2d_music_mdl_internal(
            cir_snapshots=cir_snaps,
            selected_pairs=self.virtual_pairs,
            freq_axis=selected_freq_axis, # 传入筛选后的物理频率
            imaging_grid=self.config.imaging_grid,
            max_targets=params['mdl_max_targets'],
            fov_degrees=params['fov_degrees'],
            epsilon_r=params['epsilon_r'],
            system_delay=params['system_delay'],
            is_forward_backward=params['forward_backward'],
            diag_load=params['diag_load'],
            cir_offset = offset
        )
        
        trm_image = np.fliplr(trm_image)
        
        # 4. 寻找峰值
        grid_x, grid_y = self.config.imaging_grid
        peaks = util.find_xy_peaks_from_image(trm_image, k_hat, grid_x, grid_y, threshold_rel=0.3)
        
        if self.peak_window.maxlen != params['smooth_win']:
            self.peak_window = deque(list(self.peak_window), maxlen=params['smooth_win'])
        if len(peaks) > 0 and np.max(trm_image) > params['music_threshold']:
            self.peak_window.append(peaks[0])
        else:
            self.peak_window.clear()

        # 5. Doppler
        dop_fft_n = params['doppler_fft']
        heatmap_data = np.zeros((self.config.ft_len, dop_fft_n))
        peak_tap = -1
        try:
            dop_tx = params['doppler_tx']; dop_rx = params['doppler_rx']
            tx_idx = self.config.udp_tx_list.index(dop_tx)
            rx_idx = self.config.udp_rx_list.index(dop_rx)
            if all_c.shape[3] >= dop_fft_n:
                recent = all_c[rx_idx, tx_idx, :, -dop_fft_n:]
                dop_res = np.fft.fftshift(np.fft.fft(recent, axis=1), axes=1)
                heatmap_data = np.abs(dop_res)
                p_idx = util.find_peak_indices_np(heatmap_data[:16, :], util.threshold_func, params['dop_th_start'], params['dop_th_default'])
                if len(p_idx) > 0: peak_tap = sorted(p_idx, key=lambda x: x[0])[0][0]
        except: pass

        # 6. Breathing
        val_breath = 0.0
        try:
            if all_a.shape[3] >= dop_fft_n:
                recent_a = all_a[rx_idx, tx_idx, :, -dop_fft_n:]
                val_breath, _, _, max_motion = find_breathing_feature(np.fft.fft(recent_a, axis=1), self.config.snapshot_rate, 0.15, 0.7, 0, 16)
                if self.breathing_window.maxlen != 10: # simple fixed
                    self.breathing_window = deque(list(self.breathing_window), maxlen=10)
                self.breathing_window.append(val_breath)
        except: pass

        # 7. Fusion
        smooth_x, smooth_y = 0, 0
        has_target = False
        cnt = sum(1 for v in self.breathing_window if v > params['breath_val_th'])
        is_breathing = (cnt >= params['breath_cnt_th'])
        if len(self.peak_window) == self.peak_window.maxlen and is_breathing:
            avg = np.mean(list(self.peak_window), axis=0)
            smooth_x, smooth_y = avg[0], avg[1]
            has_target = True

        return {"music_img": trm_image, "doppler_img": heatmap_data, "peak_tap": peak_tap, "has_target": has_target, "target_pos": (smooth_x, smooth_y), "breath_val": val_breath, "max_motion": max_motion, "params": params}

    def step_waveform(self):
        params = self.config.algo_params['PLOT']
        tx, rx = params['tx_pair'], params['rx_pair']
        pair = (tx, rx)
        if pair not in self.dm.snapshots_data: 
            if self.dm.snapshots_data: pair = list(self.dm.snapshots_data.keys())[0]
            else: return None
        
        key = 'raw' if params['show_raw'] else 'abs'
        buff = self.dm.snapshots_data[pair][key]
        if len(buff)==0: return None
        
        d = buff[-1]
        d_real = np.real(d); d_imag = np.imag(d)
        is_sat = np.any(d_real >= 32767) or np.any(d_imag >= 32767)
        return {"y": np.abs(d), "y_real": d_real, "y_imag": d_imag, "max_real": np.max(d_real), "is_sat": is_sat, "params": params, "pair": pair}


    def perform_ca_cfar_2d(self,power_map,params):
        """
        完善后的 2D CA-CFAR 检测函数
        :param power_map: 非相干累加后的能量图 (Range, Doppler) 
        :param params: 包含阈值和窗口大小的配置字典
        :return: 检出点的掩码 (Mask)
        """
        # 1. 获取论文定义的窗口参数 
        rt, rg = params['range_train'], params['range_guard']     # 3, 1
        dt, dg = params['doppler_train'], params['doppler_guard'] # 4, 2
        # 2. 构建卷积核以计算周围训练单元的平均功率 [cite: 264]
        # 窗口总大小: (2*rt + 2*rg + 1) x (2*dt + 2*dg + 1)
        win_r, win_d = 2*(rt + rg) + 1, 2*(dt + dg) + 1
        kernel = np.ones((win_r, win_d), dtype=float)

        # 将中心保护区域及被测单元 (CUT) 设为 0 
        kernel[rt : rt + 2*rg + 1, dt : dt + 2*dg + 1] = 0

        # 3. 计算训练单元的功率总和与均值
        num_train_cells = np.sum(kernel)
        noise_sum = ndimage.convolve(power_map, kernel, mode='constant', cval=0.0)
        noise_avg = noise_sum / num_train_cells
        # 4. 应用 15 dB 阈值进行检测判定
        threshold_linear = 10**(params['cfar_threshold_db'] / 10.0)
        detection_mask = power_map > (noise_avg * threshold_linear)

        return detection_mask, noise_avg

    def estimate_aoa(self, phase_vec, params):
        """
            修改后的 AOA 估算函数：支持 2x5 稀疏映射与 2D 角度提取
            :param phase_vec: 原始采集顺序的 8 通道复数值 (Shape: (8,))
            :return: azimuth_rad
        """
        # 1. 执行 2D 稀疏阵列映射
        # grid_2d = self.map_to_2d_grid(phase_vec)
        
        method = params.get("aoa_method", "FFT")

        # 1. 计算原始角度列表 (弧度)
        if method == "FFT":
            az_list = [-self.calculate_fft_aoa(phase_vec, params)]
        elif method == "MUSIC":
            az_list = [-self.calculate_music_aoa(phase_vec, params)]
        elif method == "Capon":
            # calculate_capon_aoa 已返回 List[float] (含多峰检测)
            az_list = [-a for a in self.calculate_capon_aoa(phase_vec, params)]
        else:
            az_list = [0.0]

        # 2. 校准: 对每个角度应用
        calib_mode = params.get('aoa_calib_mode', 'linear')
        offset_deg = params.get("aoa_offset_deg", 0.0)
        scale = params.get("aoa_scale", 1.0)
        offset_rad = np.deg2rad(offset_deg)

        result = []
        for az in az_list:
            if calib_mode in ('linear', 'both'):
                az = (az * scale) + offset_rad
            az = np.clip(az, -np.pi / 2, np.pi / 2)
            result.append(az)

        return result  # List[float]
    
    def calculate_2d_fft_aoa(self, grid_2d, params):
        """
        对 2x5 矩阵执行 2D-FFT 提取方位角与俯仰角
        """
        n_fft_az = params.get("fft_n_az", 64)   # 水平方向补零长度
        n_fft_el = params.get("fft_n_el", 32)   # 垂直方向补零长度

        # 1. 水平方向 FFT (Azimuth) - 对 5 列进行处理
        # 我们取两行的平均功率谱或对两行求和以增强 SNR
        az_fft = np.fft.fft(grid_2d, n=n_fft_az, axis=1)
        az_spectrum = np.sum(np.abs(az_fft), axis=0)
        az_peak_idx = np.argmax(np.fft.fftshift(az_spectrum))
        
        # 映射索引到方位角 (-1 到 1 对应 sin(theta))
        bins_az = np.linspace(-1, 1, n_fft_az)
        sin_az = bins_az[az_peak_idx]
        azimuth = np.arcsin(np.clip(sin_az, -0.99, 0.99))

        # 2. 垂直方向 FFT (Elevation) - 对 2 行进行处理
        # 提取每一列对应的垂直相位差
        el_fft = np.fft.fft(grid_2d, n=n_fft_el, axis=0)
        el_spectrum = np.sum(np.abs(el_fft), axis=1)
        el_peak_idx = np.argmax(np.fft.fftshift(el_spectrum))
        
        # 映射索引到俯仰角
        bins_el = np.linspace(-1, 1, n_fft_el)
        sin_el = bins_el[el_peak_idx]
        elevation = np.arcsin(np.clip(sin_el, -0.99, 0.99))

        return azimuth, elevation

    def calculate_music_2d_aoa(self, raw_vec, params):
        """
        2D MUSIC 算法实现：针对 7 个非均匀阵元提取方位角与俯仰角 
        raw_vec 顺序: [T0T0, T0T1, T0R0, T0R1, T1T0, T1T1, T1R0, T1R1]
        """
        # 1. 整理 7 个有效观测值并处理冗余点
        # 物理坐标 (x, y) 单位为 lambda
        # Row 0 (y=0): TRX0(0,0), TRX1(1,0)冗余处理, T1-T1(2,0)
        # Row 1 (y=-0.5): T0-R0(0,-0.5), T0-R1(0.5,-0.5), T1-R0(1,-0.5), T1-R1(1.5,-0.5)
        X = np.array([
            raw_vec[0],                   # (0, 0)
            (raw_vec[1] + raw_vec[4])/2,  # (1, 0) - 冗余均值 
            raw_vec[5],                   # (2, 0)
            raw_vec[2],                   # (0, -0.5)
            raw_vec[3],                   # (0.5, -0.5)
            raw_vec[6],                   # (1, -0.5)
            raw_vec[7]                    # (1.5, -0.5)
        ]).reshape(-1, 1)

        # 阵元物理坐标矩阵 (7x2)
        pos = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, -0.5], [0.5, -0.5], [1.0, -0.5], [1.5, -0.5]
        ])

        # 2. 计算协方差矩阵 R
        R = X @ X.conj().T  # 实际应用中建议对多帧进行平均以获得更稳定的 R

        # 3. 特征分解并提取噪声子空间 
        evals, evecs = np.linalg.eigh(R)
        # 假设单目标，噪声子空间为前 6 个特征向量
        En = evecs[:, :-1] 

        # 4. 2D 空间搜索
        az_range = np.deg2rad(np.linspace(-60, 60, 60)) # 方位角搜索范围
        el_range = np.deg2rad(np.linspace(-30, 30, 30)) # 俯仰角搜索范围
        
        max_val = -1
        best_az, best_el = 0, 0

        # 预计算部分常数
        EnEnH = En @ En.conj().T

        for az in az_range:
            for el in el_range:
                # 2D 导向矢量公式: a = exp(j*2pi * (x*sin(az)*cos(el) + y*sin(el)))
                # 注意：此处坐标已归一化，无需再乘 lambda [cite: 192]
                phase = 2 * np.pi * (pos[:, 0] * np.sin(az) * np.cos(el) + pos[:, 1] * np.sin(el))
                a = np.exp(1j * phase).reshape(-1, 1)
                
                # 计算伪谱分母: a' * En * En' * a
                denom = np.abs(a.conj().T @ EnEnH @ a)[0, 0]
                p_music = 1.0 / denom
                
                if p_music > max_val:
                    max_val = p_music
                    best_az, best_el = az, el
                    
        return best_az, best_el

    def calculate_fft_aoa(self, phase_vec, params):
        """
        论文推荐的 Angle FFT 算法 [cite: 270, 273]
        """
        n_fft = params.get("fft_n", 16)
        # 1. 空间维 FFT (对 8 个天线数据进行补零 FFT)
        #  提到通过空间维度的傅里叶变换可以区分方位角 (AoA)
        spatial_fft = np.fft.fft(phase_vec, n=n_fft)
        spatial_fft = np.fft.fftshift(spatial_fft)
        
        # 2. 寻找峰值索引
        mag_spectrum = np.abs(spatial_fft)
        peak_idx = np.argmax(mag_spectrum)
        
        # 3. 将索引映射到正弦空间再转角度
        # 映射范围从 -pi 到 pi 对应 sin(theta) 从 -1 到 1
        bins = np.linspace(-1, 1, n_fft)
        sin_theta = bins[peak_idx]
        
        # 裁剪范围防止 arcsin 溢出
        sin_theta = np.clip(sin_theta, -0.99, 0.99)
        return np.arcsin(sin_theta)
    
    def calculate_music_aoa(self, phase_vec, params):
        """
        增强型 MUSIC 算法 
        """
        # 由于单点只有一帧空间向量，需要构造协方差矩阵 (或使用空间平滑)
        # 这里使用简单的外积构造初步 R

        N = len(phase_vec) 
        x = phase_vec.reshape(-1, 1)
        R = x @ x.conj().T
        
        # --- [新增：前后向平滑优化] ---
        if params.get("music_forward_backward", False):
            # 创建反对角矩阵 J
            J = np.flip(np.eye(N), axis=0)
            # 应用公式: R = 0.5 * (R + J * R_conj * J)
            R = 0.5 * (R + J @ R.conj() @ J)
        
        # ----------------------------

        # 特征分解
        evals, evecs = np.linalg.eigh(R)
        # 假设信号源数量为 1
        noise_subspace = evecs[:, :-1]
        
        # 构建扫描矢量并计算伪谱
        angles = np.linspace(-np.pi/2, np.pi/2, 180)
        d = params["antenna_spacing"]
        pseudo_spectrum = []
        # 预生成天线索引序列，长度为 N
        n_indices = np.arange(N)
        for theta in angles:
            # 导向矢量 a(theta) = exp(j * 2pi * d * sin(theta) * n)
            steering_vec = np.exp(1j * 2 * np.pi * d * np.sin(theta) * n_indices)
            v = steering_vec.reshape(-1, 1)
            # 伪谱公式: 1 / (a' * En * En' * a)
            denom = v.conj().T @ noise_subspace @ noise_subspace.conj().T @ v
            pseudo_spectrum.append(1.0 / np.abs(denom[0, 0]))
        # print("angles ",angles[np.argmax(pseudo_spectrum)])
        return angles[np.argmax(pseudo_spectrum)]

    def _get_capon_antenna_x(self, params):
        """
        获取 Capon 导向矢量所需的实际天线 x 坐标 (归一化到波长).
        对齐 DBF 的 _init_dbf_steering 实现逻辑 (Dubhe 文档 Section 3.8.3).

        返回:
            ant_x:  选中通道的归一化 x 坐标 (单位: 波长)
            channels: 对应的原始 8 通道索引列表
        """
        wavelength = 2.99792458e8 / params.get('center_freq', 7.9872e9)
        # 8 通道虚拟天线 x 坐标 (m), 与 _init_dbf_steering 中的 all_virt_x 一致
        # 通道顺序: [TX1-RX4, TX1-RX5, TX1-RX6, TX1-RX7,
        #            TX2-RX4, TX2-RX5, TX2-RX6, TX2-RX7]
        all_virt_x = np.array([-0.038, 0.0, -0.038, -0.019, 0.0, 0.038, 0.0, 0.019])
        channels = params.get('indices_azimuth', [2, 3, 6, 7])
        ant_x = all_virt_x[channels] / wavelength  # 归一化到波长
        return ant_x, channels

    def calculate_capon_aoa(self, phase_vec, params):
        """
        Capon (MVDR) 自适应波束形成测角, 论文 Eq.11-13.
        对协方差矩阵求逆, 自适应抑制旁瓣, 分辨率优于 DBF/FFT.
        返回角度列表 (弧度) — 支持多峰检测.

        使用真实天线位置构建导向矢量 (替代 ULA 假设),
        支持 ant_calib_en / ant_calib_phase 通道相位校准.
        校准方式与 DBF 的 _init_dbf_steering 对齐: 校准量作用于引导矢量.
        """
        N = len(phase_vec)
        diag_load = params.get("capon_diag_load", 1e-3)

        # --- 预计算引导矢量的相位校准因子 (对齐 DBF / Dubhe Section 3.8.3) ---
        ant_x, channels = self._get_capon_antenna_x(params)
        if params.get('ant_calib_en', False):
            calib_phase = np.array(params.get('ant_calib_phase',
                                              [0.0] * 8), dtype=np.float64)
            calib_sv = np.exp(1j * calib_phase[channels]).reshape(-1, 1)
        else:
            calib_sv = np.ones((N, 1), dtype=np.complex64)

        # 协方差矩阵 (单快照)
        x = phase_vec.reshape(-1, 1)
        R = x @ x.conj().T

        # 前后向平滑
        if params.get("music_forward_backward", False):
            J = np.flip(np.eye(N), axis=0)
            R = 0.5 * (R + J @ R.conj() @ J)

        # 对角加载保证可逆
        R += np.eye(N) * diag_load * np.abs(np.trace(R))

        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            return [0.0]

        # 角度扫描 — 使用真实天线位置 + 引导矢量相位校准
        angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
        spectrum = np.zeros(len(angles))
        for i, theta in enumerate(angles):
            sv = np.exp(1j * 2 * np.pi * ant_x * np.sin(theta)).reshape(-1, 1)
            sv = sv * calib_sv  # 相位校准作用于引导矢量 (与 DBF 一致)
            spectrum[i] = 1.0 / np.real(sv.conj().T @ R_inv @ sv)

        # 多峰检测
        return self._find_angle_peaks(spectrum, angles, params)

    def step_point_cloud(self):
        """
        点云生成算法：Shape 推导与 CA-CFAR 处理
        """
        # 1. 获取输入快照: (N_rx, N_tx, RangeBins, Snapshots) -> (4, 2, 32, 128)
        all_c = self.dm.get_all_snapshot_as_array('complex')
        all_a = self.dm.get_all_snapshot_as_array('abs')
        if all_c is None: return None

        params = self.config.algo_params['POINT-CLOUD']
        N_snaps = params['snapshots'] # 使用论文中的 32 个快照
        cir_offset = params['cir_offset']
        current_cube = all_c[:,:,cir_offset:, -N_snaps:]
        #获取的数据已经完成静态杂波处理
        # 2. 静态杂波去除 (Static Clutter Removal)
        cube_no_static = current_cube
        # 3. 多普勒 FFT (Doppler FFT)
        # 对快照维度进行 FFT, # Shape: (N_rx, N_tx, Range, Doppler)
        rd_cube = np.fft.fft(cube_no_static, axis=3)
        # --- 核心修改：统一天线映射逻辑 ---
        # 将天线排列为 [TX0RX0, TX0RX1, TX0RX2, TX0RX3, TX1RX0, ...] 的 8 通道形式
        # 这里使用 transpose(1, 0, 2, 3) 将其变为 (TX, RX, R, D)，然后 reshape
        rd_cube_flat = rd_cube.transpose(1, 0, 2, 3).reshape(8, rd_cube.shape[2], rd_cube.shape[3])
        valid_indices = params.get('indices_azimuth', [0, 1, 2, 3, 4, 5, 6, 7])
        # 4. 非相干累加得到能量图 (Power Map)
        # 根据开关决定 CFAR 使用哪些天线的能量
        if params.get('cfar_only_selected', False):
            # 仅使用选中的天线进行能量叠加
            rd_selected_for_cfar = rd_cube_flat[valid_indices, :, :]
            power_map = np.sum(np.abs(rd_selected_for_cfar)**2, axis=0)
        else:
            # 使用全部 8 个天线的能量叠加（默认行为）
            power_map = np.sum(np.abs(rd_cube_flat)**2, axis=0)
        # 5. CA-CFAR 检测逻辑
        mask, noise_avg = self.perform_ca_cfar_2d(power_map, params)      
       # 2. 提取检出点的索引
        hit_indices = np.argwhere(mask) # 返回 [[r1, d1], [r2, d2], ...]                
        
        detected_points = []
        for r_idx, d_idx in hit_indices:
            # 3. 提取该点在 8 个通道的相位特征向量
            # Shape: (4, 2) -> (8,)
            # --- 测角部分：始终只使用选中的天线 ---
            # 直接从 rd_cube_flat 中提取对应的 8 通道相位，然后切片
            phase_vec_full = rd_cube_flat[:, r_idx, d_idx]
            phase_vec_selected = phase_vec_full[valid_indices]

            # AoA 角度估算 → 返回 List[float]
            az_list = self.estimate_aoa(phase_vec_selected, params)
            # 5. 坐标转换与点云构建
            # Range = 索引 * 分辨率
            dist = (r_idx + cir_offset - 6) * params['dist_per_tap']
            snr_val = 10 * np.log10(power_map[r_idx, d_idx] / (noise_avg[r_idx, d_idx] + 1e-12))

            for az_angle in az_list:
                # 笛卡尔坐标映射: x(横向), y(纵向)
                point_x = dist * np.sin(az_angle)
                point_y = -dist * np.cos(az_angle)  # 纵向深度
                detected_points.append({
                    'pos': (point_x, point_y),
                    'snr': snr_val,
                    'time': time.time()  # 记录时间戳用于"出现后消失"效果
                })

        # 6. Breathing
        dop_fft_n = params['snapshots']
        val_breath = 0.0
        try:
            if all_a.shape[3] >= dop_fft_n:
                dop_tx = params['doppler_tx']; dop_rx = params['doppler_rx']
                tx_idx = self.config.udp_tx_list.index(dop_tx)
                rx_idx = self.config.udp_rx_list.index(dop_rx)
                recent_a = all_a[rx_idx, tx_idx, :, -dop_fft_n:]
                val_breath, _, _, max_motion = find_breathing_feature(np.fft.fft(recent_a, axis=1), self.config.snapshot_rate, 0.15, 0.7, 0, 16)
                if self.breathing_window.maxlen != 10: # simple fixed
                    self.breathing_window = deque(list(self.breathing_window), maxlen=10)
                self.breathing_window.append(val_breath)
        except: pass
        val_breath = float(val_breath) # 强制转换
        return {
        "detected_points": detected_points, # 包含 pos, time, snr 的列表
        "params": self.config.algo_params['POINT-CLOUD'],
        "breath_val": val_breath,       # <--- 新增：返回呼吸数值
        }

    def step_point_cloud_optimized(self):
        """
        优化版点云算法: 加窗 + DC去除 + leakage roll + 速度/距离门限 + 峰值滤波 + 子网格细化
        AoA 支持 FFT / MUSIC / DBF 三种方式
        """
        all_c = self.dm.get_all_snapshot_as_array('complex')
        all_a = self.dm.get_all_snapshot_as_array('abs')
        if all_c is None:
            return None

        params = self.config.algo_params['POINT-CLOUD-OPTIMIZED']
        N_snaps = params['snapshots']
        cir_comb = params.get('cir_combine_num', 1)
        leakage_offset = params['leakage_offset']
        current_cube = all_c[:, :, :, -N_snaps:]  # (4, 2, 32, N)

        # 1. CIR 相干积累: 每 cir_comb 帧复数累加取均值
        #    → 提升 SNR ~10log10(cir_comb) dB
        #    → 压缩慢时间轴, 等效帧周期 × cir_comb (抗混叠低通)
        #    → 观测窗口不变时, 等效 Doppler bin 更少但 bin 内噪底更低
        if cir_comb > 1:
            n_comb = current_cube.shape[3] // cir_comb
            trim = n_comb * cir_comb
            current_cube = current_cube[:, :, :, :trim] \
                .reshape(4, 2, 32, n_comb, cir_comb).mean(axis=4)
            # shape: (4, 2, 32, n_comb)

        # 2. 泄漏处理: np.roll 替代截断
        current_cube = np.roll(current_cube, -leakage_offset, axis=2)

        # # 3. 慢时间 DC 去除
        # if params.get('doppler_dc_remove', True):
        #     current_cube = current_cube - np.mean(current_cube, axis=3, keepdims=True)

        # 4. 慢时间加窗 (Chebyshev)
        if params.get('doppler_window') == 'chebyshev':
            atten = params.get('doppler_win_atten', 60)
            win = chebwin(current_cube.shape[3], at=atten)
            current_cube = current_cube * win[np.newaxis, np.newaxis, np.newaxis, :]

        # 5. Doppler FFT
        rd_cube = np.fft.fft(current_cube, axis=3)

        # 6. 展平 8 通道 + 选通道 + 非相干合并
        rd_cube_flat = rd_cube.transpose(1, 0, 2, 3).reshape(
            8, rd_cube.shape[2], rd_cube.shape[3])
        valid_indices = params.get('indices_azimuth', [2, 3, 6, 7])
        if params.get('cfar_only_selected', True):
            rd_sel = rd_cube_flat[valid_indices, :, :]
        else:
            rd_sel = rd_cube_flat
        power_map = np.sum(np.abs(rd_sel) ** 2, axis=0)

        # 7. CA-CFAR
        mask, noise_avg = self.perform_ca_cfar_2d(power_map, params)

        # 8. Velocity + Range gating (仿 Dubhe 双侧通带)
        n_dop = power_map.shape[1]
        v_min = params.get('cfar_velocity_min', 1)
        v_max = params.get('cfar_velocity_max', 20)
        vel_mask = np.zeros(n_dop, dtype=bool)
        vel_mask[v_min:v_max] = True
        vel_mask[n_dop - v_max:n_dop - v_min] = True
        mask &= vel_mask[np.newaxis, :]

        r_min = params.get('cfar_range_min', 2)
        r_max = params.get('cfar_range_max', 20)
        mask[:r_min, :] = False
        mask[r_max:, :] = False

        # 9. 局部峰值滤波
        if params.get('cfar_doppler_peak_flag'):
            mask &= (power_map == ndimage.maximum_filter(power_map, size=(1, 3)))
        if params.get('cfar_range_peak_flag'):
            mask &= (power_map == ndimage.maximum_filter(power_map, size=(3, 1)))

        # 10. 检出 + 子网格细化 + AoA
        hit_indices = np.argwhere(mask)
        if params.get('aoa_debug_print', False) and len(hit_indices) > 0:
            print(f"\n[CFAR DEBUG] 检出 {len(hit_indices)} 个 range-doppler bin:")
            for r_idx, d_idx in hit_indices:
                snr = 10 * np.log10(power_map[r_idx, d_idx] / (noise_avg[r_idx, d_idx] + 1e-12))
                dist = r_idx * params['dist_per_tap']
                print(f"  range_bin={r_idx} ({dist:.2f}m), doppler_bin={d_idx}, SNR={snr:.1f}dB")
        detected_points = []
        for r_idx, d_idx in hit_indices:
            r_fine, d_fine = r_idx, d_idx
            if params.get('subbin_refine_en', True):
                r_fine, d_fine = self._subbin_refine(power_map, r_idx, d_idx)

            phase_vec_full = rd_cube_flat[:, r_idx, d_idx]
            phase_vec_sel = phase_vec_full[valid_indices]

            # AoA 分发: FFT / MUSIC / DBF → 每个返回 List[float]
            aoa_method = params.get('aoa_method', 'MUSIC')
            if aoa_method == 'DBF':
                if getattr(self, '_dbf_sv', None) is None:
                    self._init_dbf_steering(params)
                az_list = self._dbf_estimate(phase_vec_sel, params)
            else:
                az_list = self.estimate_aoa(phase_vec_sel, params)

            dist = r_fine * params['dist_per_tap']
            snr_val = 10 * np.log10(
                power_map[r_idx, d_idx] / (noise_avg[r_idx, d_idx] + 1e-12))
            for az_angle in az_list:
                point_x = dist * np.sin(az_angle)
                point_y = -dist * np.cos(az_angle)
                detected_points.append({
                    'pos': (-point_x, point_y),
                    'snr': snr_val,
                    'time': time.time()
                })

        # 11. Breathing
        dop_fft_n = params['snapshots']
        val_breath = 0.0
        try:
            if all_a.shape[3] >= dop_fft_n:
                dop_tx = params['doppler_tx']; dop_rx = params['doppler_rx']
                tx_idx = self.config.udp_tx_list.index(dop_tx)
                rx_idx = self.config.udp_rx_list.index(dop_rx)
                recent_a = all_a[rx_idx, tx_idx, :, -dop_fft_n:]
                val_breath, _, _, max_motion = find_breathing_feature(
                    np.fft.fft(recent_a, axis=1),
                    self.config.snapshot_rate, 0.15, 0.7, 0, 16)
                if self.breathing_window.maxlen != 10:
                    self.breathing_window = deque(list(self.breathing_window), maxlen=10)
                self.breathing_window.append(val_breath)
        except:
            pass
        val_breath = float(val_breath)
        return {
            "detected_points": detected_points,
            "params": self.config.algo_params['POINT-CLOUD-OPTIMIZED'],
            "breath_val": val_breath,
        }

    def step_angle_spectrum_view(self):
        """
        角度谱调试视图: 与 step_point_cloud_optimized 完全相同的预处理 + CFAR,
        但对每个检出 bin 不做 argmax/寻峰, 而是返回完整的 DBF 角度谱曲线.

        用于调试: 直观看同一个 range-doppler bin 的角度谱上有没有第二个峰.
        """
        all_c = self.dm.get_all_snapshot_as_array('complex')
        all_a = self.dm.get_all_snapshot_as_array('abs')
        if all_c is None:
            return None

        params = self.config.algo_params['POINT-CLOUD-OPTIMIZED']
        N_snaps = params['snapshots']
        cir_comb = params.get('cir_combine_num', 1)
        leakage_offset = params['leakage_offset']
        current_cube = all_c[:, :, :, -N_snaps:]

        # 1-5. 与 step_point_cloud_optimized 完全相同的预处理
        if cir_comb > 1:
            n_comb = current_cube.shape[3] // cir_comb
            trim = n_comb * cir_comb
            current_cube = current_cube[:, :, :, :trim] \
                .reshape(4, 2, 32, n_comb, cir_comb).mean(axis=4)
        current_cube = np.roll(current_cube, -leakage_offset, axis=2)
        # if params.get('doppler_dc_remove', True):
        #     current_cube = current_cube - np.mean(current_cube, axis=3, keepdims=True)
        if params.get('doppler_window') == 'chebyshev':
            atten = params.get('doppler_win_atten', 60)
            win = chebwin(current_cube.shape[3], at=atten)
            current_cube = current_cube * win[np.newaxis, np.newaxis, np.newaxis, :]
        rd_cube = np.fft.fft(current_cube, axis=3)

        # 6. 展平 + 选通道 + 非相干合并
        rd_cube_flat = rd_cube.transpose(1, 0, 2, 3).reshape(
            8, rd_cube.shape[2], rd_cube.shape[3])
        valid_indices = params.get('indices_azimuth', [2, 3, 6, 7])
        if params.get('cfar_only_selected', True):
            rd_sel = rd_cube_flat[valid_indices, :, :]
        else:
            rd_sel = rd_cube_flat
        power_map = np.sum(np.abs(rd_sel) ** 2, axis=0)

        # 7. CA-CFAR (与 step_point_cloud_optimized 相同)
        mask, noise_avg = self.perform_ca_cfar_2d(power_map, params)
        n_dop = power_map.shape[1]
        v_min = params.get('cfar_velocity_min', 1)
        v_max = params.get('cfar_velocity_max', 20)
        vel_mask = np.zeros(n_dop, dtype=bool)
        vel_mask[v_min:v_max] = True
        vel_mask[n_dop - v_max:n_dop - v_min] = True
        mask &= vel_mask[np.newaxis, :]
        r_min = params.get('cfar_range_min', 2)
        r_max = params.get('cfar_range_max', 20)
        mask[:r_min, :] = False
        mask[r_max:, :] = False
        if params.get('cfar_doppler_peak_flag'):
            mask &= (power_map == ndimage.maximum_filter(power_map, size=(1, 3)))
        if params.get('cfar_range_peak_flag'):
            mask &= (power_map == ndimage.maximum_filter(power_map, size=(3, 1)))

        # 8. 初始化 DBF 引导矢量
        if getattr(self, '_dbf_sv', None) is None:
            self._init_dbf_steering(params)
        angles_deg = np.rad2deg(self._dbf_angles)

        # 9. 对每个 CFAR 检出 bin, 计算完整角度谱
        hit_indices = np.argwhere(mask)
        spectra = []
        for r_idx, d_idx in hit_indices:
            r_fine, d_fine = r_idx, d_idx
            if params.get('subbin_refine_en', True):
                r_fine, d_fine = self._subbin_refine(power_map, r_idx, d_idx)

            phase_vec_full = rd_cube_flat[:, r_idx, d_idx]
            phase_vec_sel = phase_vec_full[valid_indices]

            # DBF 角度谱 (完整曲线, 不做 argmax)
            x = phase_vec_sel.reshape(-1, 1)
            pwr = np.abs(self._dbf_sv.conj().T @ x) ** 2
            pwr = pwr.flatten()

            dist = r_fine * params['dist_per_tap']
            snr_db = 10 * np.log10(power_map[r_idx, d_idx] / (noise_avg[r_idx, d_idx] + 1e-12))

            spectra.append({
                'angles_deg': angles_deg,
                'pwr_linear': pwr,
                'pwr_db': 10 * np.log10(pwr + 1e-12),
                'range_m': dist,
                'doppler_bin': d_idx,
                'snr_db': snr_db,
            })

        return {
            'spectra': spectra,
            'params': params,
            'power_map': power_map,
            'n_detections': len(spectra),
        }

    def step_angle_spectrum_raw(self):
        """
        Raw DBF angle spectrum for the first few range bins — no CFAR filtering.

        与 step_angle_spectrum_view 共享完全相同的预处理, 但跳过 CFAR 检测.
        对前几个 range bin, 各取其最强 doppler bin 的相位向量做 DBF 测角,
        返回每条 range bin 的完整角度谱曲线 (power vs. angle).

        用途: 观察早期 range bin 的原始 DBF 角度谱形态, 理解无 CFAR 筛选时的测角数据.
        """
        all_c = self.dm.get_all_snapshot_as_array('complex')
        if all_c is None:
            return None

        params = self.config.algo_params['POINT-CLOUD-OPTIMIZED']
        N_snaps = params['snapshots']
        cir_comb = params.get('cir_combine_num', 1)
        leakage_offset = params['leakage_offset']
        current_cube = all_c[:, :, :, -N_snaps:]

        # 1-5. 与 step_point_cloud_optimized 完全相同的预处理
        if cir_comb > 1:
            n_comb = current_cube.shape[3] // cir_comb
            trim = n_comb * cir_comb
            current_cube = current_cube[:, :, :, :trim] \
                .reshape(4, 2, 32, n_comb, cir_comb).mean(axis=4)
        current_cube = np.roll(current_cube, -leakage_offset, axis=2)
        # if params.get('doppler_dc_remove', True):
        #     current_cube = current_cube - np.mean(current_cube, axis=3, keepdims=True)
        if params.get('doppler_window') == 'chebyshev':
            atten = params.get('doppler_win_atten', 60)
            win = chebwin(current_cube.shape[3], at=atten)
            current_cube = current_cube * win[np.newaxis, np.newaxis, np.newaxis, :]
        rd_cube = np.fft.fft(current_cube, axis=3)

        # 6. 展平 + 选通道
        rd_cube_flat = rd_cube.transpose(1, 0, 2, 3).reshape(
            8, rd_cube.shape[2], rd_cube.shape[3])
        valid_indices = params.get('indices_azimuth', [2, 3, 6, 7])
        rd_sel = rd_cube_flat[valid_indices, :, :]  # [n_ch, n_range, n_dop]

        # 初始化 DBF 引导矢量
        if getattr(self, '_dbf_sv', None) is None:
            self._init_dbf_steering(params)
        angles_deg = np.rad2deg(self._dbf_angles)

        # 非相干合并功率图 (用于找最强 doppler bin)
        power_map = np.sum(np.abs(rd_sel) ** 2, axis=0)  # [n_range, n_dop]

        # 选定前几个 range bin (可配置)
        r_start = params.get('asr_range_start', 2)
        n_range_bins = params.get('asr_n_range_bins', 8)
        r_end = min(r_start + n_range_bins, rd_sel.shape[1])

        # 对每个 range bin: 取最强 doppler bin, 做 DBF 角度谱
        spectra = []
        for r_idx in range(r_start, r_end):
            d_idx = np.argmax(power_map[r_idx, :])  # 最强 doppler bin

            phase_vec = rd_sel[:, r_idx, d_idx]
            x = phase_vec.reshape(-1, 1)
            pwr = np.abs(self._dbf_sv.conj().T @ x) ** 2
            pwr = pwr.flatten()

            # 亚 bin 精炼 (可选)
            r_fine = r_idx
            if params.get('subbin_refine_en', True):
                r_fine, _ = self._subbin_refine(power_map, r_idx, d_idx)

            dist = r_fine * params['dist_per_tap']
            peak_angle = angles_deg[np.argmax(pwr)]

            spectra.append({
                'angles_deg': angles_deg,
                'pwr_linear': pwr,
                'pwr_db': 10 * np.log10(pwr + 1e-12),
                'range_m': dist,
                'range_bin': r_idx,
                'doppler_bin': int(d_idx),
                'peak_angle': peak_angle,
            })

        return {
            'spectra': spectra,
            'angles_deg': angles_deg,
            'power_map': 10 * np.log10(power_map + 1e-12),
            'params': params,
            'n_range': len(spectra),
        }

    def step_point_cloud_ra_cfar(self):
        """
        Range-Azimuth Two-Pass CFAR (论文 IEEE JSEN 2024 Algorithm 1):
          1. 计算 DBF Range-Azimuth 热力图
          2. Pass 1: 沿 Range 维 1D CFAR
          3. Pass 2: 沿 Azimuth 维 1D CFAR (仅对 Pass1 检出的 range)
          4. 每个检出 (r, θ) → 笛卡尔坐标 → 点云

        与 step_point_cloud_optimized 的核心区别:
          - 角度作为检测维度参与 CFAR, 而非事后测角
          - 不使用 Doppler 维 (每 range 取最强 Doppler 代替)
        """
        params = self.config.algo_params['RA-CFAR']

        # 1. 计算 Range-Azimuth 热力图 (Capon, 含 R^{-1} 和 chirp 原始数据)
        H, ranges_m, angles_deg, A_all, R_inv_list = self._compute_ra_heatmap(params)
        if H is None:
            return None

        # 2. CFAR 检测
        cfar_mode = params.get('cfar_mode', '2d_ca')
        if cfar_mode == '2d_ca':
            # 2D CA-CFAR: 复用 perform_ca_cfar_2d, azimuth 映射到 doppler 维
            cfar_params = params.copy()
            cfar_params['doppler_train'] = params.get('azimuth_train', 4)
            cfar_params['doppler_guard'] = params.get('azimuth_guard', 2)
            mask, noise_avg = self.perform_ca_cfar_2d(H, cfar_params)
            # 峰值滤波
            if params.get('cfar_range_peak_flag', False):
                mask &= (H == ndimage.maximum_filter(H, size=(3, 1)))
            if params.get('cfar_azimuth_peak_flag', False):
                mask &= (H == ndimage.maximum_filter(H, size=(1, 3)))
            # Range 门限
            r_min, r_max = params.get('cfar_range_min', 2), params.get('cfar_range_max', 20)
            mask[:r_min, :] = False
            mask[r_max:, :] = False

            hit_indices = np.argwhere(mask)
            detected_points = []
            for r_idx, a_idx in hit_indices:
                dist = ranges_m[r_idx]
                az_deg = angles_deg[a_idx]
                calib_mode = params.get('aoa_calib_mode', 'linear')
                if calib_mode in ('linear', 'both'):
                    az_deg = az_deg * params.get('aoa_scale', 1.0) + params.get('aoa_offset_deg', 0.0)
                az_rad = np.deg2rad(np.clip(az_deg, -90, 90))
                point_x = dist * np.sin(az_rad)
                point_y = -dist * np.cos(az_rad)
                snr = 10 * np.log10(H[r_idx, a_idx] / (noise_avg[r_idx, a_idx] + 1e-12))
                detected_points.append({
                    'pos': (-point_x, point_y),
                    'snr': snr,
                    'time': time.time(),
                    'range_idx': int(r_idx),
                    'azimuth_idx': int(a_idx),
                })
        else:  # 'two_pass'
            detected_points = self._two_pass_cfar(H, params)

        # 3. 每个检出点计算速度 (论文 Eq.15)
        for pt in detected_points:
            k = pt['range_idx']
            i = pt['azimuth_idx']
            sv = self._capon_sv[:, i].reshape(-1, 1)
            R_inv = R_inv_list[k]
            if R_inv is not None:
                w = R_inv @ sv / np.real(sv.conj().T @ R_inv @ sv)
                y = w.conj().T @ A_all[:, k, :]
            else:
                y = sv.conj().T @ A_all[:, k, :]
            y = y.flatten()
            L_chirps = len(y)
            dop_spec = np.abs(np.fft.fft(y)) ** 2
            half = L_chirps // 2
            d_peak = np.argmax(dop_spec[:half])
            snapshot_rate = self.config.snapshot_rate
            wavelength = 2.99792458e8 / params['center_freq']
            v_res = wavelength * snapshot_rate / (2.0 * L_chirps)
            pt['velocity'] = float(d_peak * v_res)

        # 4. Breathing
        val_breath = 0.0
        try:
            all_a = self.dm.get_all_snapshot_as_array('abs')
            if all_a is not None and all_a.shape[3] >= params['snapshots']:
                dop_tx = params.get('doppler_tx', 1)
                dop_rx = params.get('doppler_rx', 6)
                tx_idx = self.config.udp_tx_list.index(dop_tx)
                rx_idx = self.config.udp_rx_list.index(dop_rx)
                recent_a = all_a[rx_idx, tx_idx, :, -params['snapshots']:]
                val_breath, _, _, _ = find_breathing_feature(
                    np.fft.fft(recent_a, axis=1),
                    self.config.snapshot_rate, 0.15, 0.7, 0, 16)
                if self.breathing_window.maxlen != 10:
                    self.breathing_window = deque(list(self.breathing_window), maxlen=10)
                self.breathing_window.append(val_breath)
        except:
            pass

        return {
            "detected_points": detected_points,
            "params": params,
            "breath_val": float(val_breath),
        }

    def step_ra_heatmap_view(self):
        """
        Range-Azimuth 热力图可视化 (Capon):
          与 RA-CFAR 共用参数和 _compute_ra_heatmap, 但不做 CFAR 检测.
          返回热力图数据供 plot_panel 显示为 imshow.
        """
        params = self.config.algo_params['RA-CFAR']
        H, ranges_m, angles_deg, _, _ = self._compute_ra_heatmap(params)
        if H is None:
            return None
        # dB 尺度更直观
        H = H -np.sqrt(np.sum(np.square(H)) / (H.shape[0] * H.shape[1]))
        H_db = 10 * np.log10(H + 1e-12)
        return {
            'heatmap': H,
            'ranges_m': ranges_m,
            'angles_deg': angles_deg,
            'params': params,
        }

    def _seat_ra_energy(self, H, ranges_m, angles_deg, seat):
        """
        向量化: 计算 RA 热力图像素中落在座位椭圆内的能量和.

        坐标映射: x = r*sin(θ),  y = -r*cos(θ)
        椭圆判断: ((x-cx)/rx)² + ((y-cy)/ry)² < 1
        """
        r_2d = ranges_m[:, np.newaxis]          # (K, 1)
        a_2d = np.deg2rad(angles_deg[np.newaxis, :])  # (1, I)
        x = r_2d * np.sin(a_2d)                 # (K, I)
        y = -r_2d * np.cos(a_2d)                # (K, I)
        xn = (x - seat['cx']) / seat['rx']
        yn = (y - seat['cy']) / seat['ry']
        mask = (xn * xn + yn * yn) < 1.0
        if not np.any(mask):
            return 0.0
        return float(np.max(H[mask]))

    def _ra_to_cartesian(self, H, ranges_m, angles_deg,
                          x_range=(-3, 3), y_range=(-6, 0), res=0.05):
        """
        RA 热力图 polar→Cartesian 重映射.
        H: (n_range, n_angle) 线性功率
        返回: H_cart (ny, nx), xs (nx,), ys (ny,)
        """
        interp = RegularGridInterpolator(
            (ranges_m, angles_deg), H,
            bounds_error=False, fill_value=np.nan)

        nx = max(1, int(round((x_range[1] - x_range[0]) / res)) + 1)
        ny = max(1, int(round((y_range[1] - y_range[0]) / res)) + 1)
        xs = np.linspace(x_range[0], x_range[1], nx)
        ys = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(xs, ys)  # (ny, nx)

        R = np.sqrt(X**2 + Y**2)
        A = np.rad2deg(np.arctan2(X, -Y))

        pts = np.stack([R.ravel(), A.ravel()], axis=1)
        H_cart = interp(pts).reshape(len(ys), len(xs))
        return H_cart, xs, ys

    def step_ra_occupancy(self):
        """
        RA热力图占用检测流水线 (无 CFAR, 无点云):
          1. 计算 Capon RA 热力图 H (复用 _compute_ra_heatmap)
          2. 背景减除: H_bg = max(H - mean(H²), 0)
          3. 每座位椭圆内能量求和
          4. 返回热力图 + 能量字典, 供 App 层做双条件判决+时序平滑
        """
        params = self.config.algo_params['RA-OCCUPANCY']
        H, ranges_m, angles_deg, _, _ = self._compute_ra_heatmap(params)
        if H is None:
            return None
        # H = np.fliplr(H)
        # 背景减除: H_bg = H - mean(H²), clip to >=0
        H_sq_mean = np.sqrt(np.sum(np.square(H)) / (H.shape[0] * H.shape[1]))
        # H_sq_mean = np.sum(H) / (H.shape[0] * H.shape[1])
        H_bg = H - H_sq_mean

        # 卷积平滑: 用均值核做 2D 卷积, 保持图像尺寸不变
        kernel_size = params.get('smooth_kernel', [2, 4])
        smooth_kernel = np.ones((kernel_size[0], kernel_size[1])) / (kernel_size[0] * kernel_size[1])
        H_bg = ndimage.convolve(H_bg, smooth_kernel, mode='reflect')

        # Read range and angle directly from the polar RA heatmap before
        # Cartesian remapping introduces its own spatial quantization.
        peak_range_idx, peak_angle_idx = np.unravel_index(
            np.nanargmax(H_bg), H_bg.shape)
        peak_range_m = float(ranges_m[peak_range_idx])
        peak_angle_deg = float(angles_deg[peak_angle_idx])

        # 每座位能量和
        occ_params = self.config.algo_params.get('SEAT-OCCUPANCY', {})
        seat_type = occ_params.get('seat_type', '4_seats')
        key = 'seats_4' if seat_type == '4_seats' else 'seats_5'
        seat_defs = occ_params.get(key, occ_params.get('seats_4', []))

        energy_dict = {}
        for seat in seat_defs:
            name = seat['name']
            main_e = self._seat_ra_energy(H_bg, ranges_m, angles_deg, seat.get('main', seat))
            special_e = self._seat_ra_energy(H_bg, ranges_m, angles_deg, seat.get('child_special', seat))
            energy_dict[name] = {'main': main_e, 'child_special': special_e}

        # Cartesian 重映射 (直接使用背景减除后的 H_bg 线性功率)
        xlim = params.get('plot_xlim', 1.5)
        y_min = params.get('plot_ylim_min', -6.0)
        y_max = params.get('plot_ylim_max', -0.1)
        H_cart, xs_cart, ys_cart = self._ra_to_cartesian(
            H_bg, ranges_m, angles_deg,
            x_range=(-xlim, xlim), y_range=(y_min, y_max), res=0.05)

        # Cartesian heatmap peak, retained separately from the polar RA peak.
        iy, ix = np.unravel_index(np.nanargmax(H_cart), H_cart.shape)
        peak_x_m = float(xs_cart[ix])
        peak_y_m = float(ys_cart[iy])

        return {
            'heatmap': H_cart,
            'h_raw': H,
            'h_bg': H_bg,
            'xs_cart': xs_cart,
            'ys_cart': ys_cart,
            'ranges_m': ranges_m,
            'angles_deg': angles_deg,
            'peak_range_m': peak_range_m,
            'peak_angle_deg': peak_angle_deg,
            'peak_x_m': peak_x_m,
            'peak_y_m': peak_y_m,
            'energy': energy_dict,
            'params': params,
        }

    # ===== Multipass CFAR (IEEE Sensors Journal 2024) helpers =====

    def perform_1d_cfar_min(self, signal, train, guard, threshold_db, ave_pad=None):
        """
        论文 Algorithm 1 的 1D CFAR: min(μ_L, μ_R) 噪声估计器

        参数:
            signal: 1D numpy array, 输入信号
            train: N_W, 训练单元半窗大小
            guard: N_G, 保护单元半窗大小
            threshold_db: γ, 阈值因子 (dB)
            ave_pad: N_ave, 边界padding均值点数, 默认等于 train

        返回:
            mask: bool array, 检出点掩码
            noise_est: float array, 每个bin的噪声估计
        """
        N = len(signal)
        mask = np.zeros(N, dtype=bool)
        noise_est = np.zeros(N, dtype=float)
        if ave_pad is None:
            ave_pad = train

        threshold_linear = 10 ** (threshold_db / 10.0)
        wg = train + guard  # 半窗+保护 = 到训练区边缘的距离

        for k in range(N):
            # 左侧训练单元范围: [k - wg - train, k - wg - 1]
            lo_L = max(0, k - wg - train)
            hi_L = max(0, k - wg)
            # 右侧训练单元范围: [k + wg + 1, k + wg + train]
            lo_R = min(N, k + wg + 1)
            hi_R = min(N, k + wg + train + 1)

            left_cells = signal[lo_L:hi_L]
            right_cells = signal[lo_R:hi_R]

            if len(left_cells) == 0 and len(right_cells) == 0:
                noise_est[k] = 1e-12
                continue

            mu_L = np.mean(left_cells) if len(left_cells) > 0 else np.inf
            mu_R = np.mean(right_cells) if len(right_cells) > 0 else np.inf
            noise_est[k] = min(mu_L, mu_R)

            if signal[k] > noise_est[k] * threshold_linear:
                mask[k] = True

        return mask, noise_est

    def _bartlett_spectrum(self, phase_vec, angles_rad):
        """Bartlett (conventional) beamforming 角度谱 — 用于 Pass 2 粗检测"""
        N = len(phase_vec)
        d = 0.5  # 天线间距 (λ)
        n_idx = np.arange(N)
        spectrum = np.zeros(len(angles_rad), dtype=float)
        for i, theta in enumerate(angles_rad):
            sv = np.exp(1j * 2 * np.pi * d * np.sin(theta) * n_idx)
            spectrum[i] = np.abs(sv.conj() @ phase_vec) ** 2
        return spectrum

    def _compute_angle_response(self, phase_vec, params, angles_rad):
        """
        统一角度响应计算: 根据 aoa_method 分发到 FFT / DBF / MUSIC

        返回: 1D array, 每个 angle_rad 对应的响应功率/谱值
        """
        method = params.get('aoa_method', 'MUSIC')
        if method == 'FFT':
            return self._fft_angle_response(phase_vec, params, angles_rad)
        elif method == 'DBF':
            return self._dbf_angle_response(phase_vec, angles_rad)
        elif method == 'Capon':
            return self._capon_angle_response(phase_vec, params, angles_rad)
        else:  # MUSIC
            return self._music_angle_response(phase_vec, params, angles_rad)

    def _angle_cfar_detect(self, spectrum, angles, params):
        """
        角度维 1D CA-CFAR: 沿角度维度滑窗, 用局部训练单元估计噪声,
        对每个角度 bin 判断是否超过 CFAR 阈值.

        参数:
            spectrum: 1D array, 角度响应功率谱 (线性值)
            angles:   1D array, 对应的角度值 (弧度)
            params:   算法参数字典

        返回:
            List[float]: 检出的角度列表 (弧度), 按功率降序排列
        """
        N = len(spectrum)
        guard = params.get('aoa_cfar_guard', 4)
        train = params.get('aoa_cfar_train', 8)
        threshold_db = params.get('aoa_cfar_threshold_db', 10.0)
        threshold_linear = 10 ** (threshold_db / 10.0)

        # 对每个角度 bin 做 CFAR
        mask = np.zeros(N, dtype=bool)
        for k in range(N):
            # 左侧训练单元: [k - guard - train, k - guard)
            lo_L = max(0, k - guard - train)
            hi_L = max(0, k - guard)
            # 右侧训练单元: [k + guard + 1, k + guard + train + 1)
            lo_R = min(N, k + guard + 1)
            hi_R = min(N, k + guard + train + 1)

            left_cells = spectrum[lo_L:hi_L]
            right_cells = spectrum[lo_R:hi_R]
            if len(left_cells) + len(right_cells) == 0:
                continue

            noise_est = (np.sum(left_cells) + np.sum(right_cells)) / (
                len(left_cells) + len(right_cells))
            if noise_est < 1e-12:
                noise_est = 1e-12

            if spectrum[k] > noise_est * threshold_linear:
                mask[k] = True

        # 合并相邻检出 (属于同一个主瓣) — 取每个连续段中功率最大的
        if not np.any(mask):
            return [angles[np.argmax(spectrum)]]  # 回退到最强峰

        # 按功率排序, 把最强峰放前面
        hit_indices = np.argwhere(mask).flatten()
        hit_powers = spectrum[hit_indices]
        sorted_idx = hit_indices[np.argsort(hit_powers)[::-1]]

        # 去重: 合并间距 < guard 的检出
        kept = [sorted_idx[0]]
        for idx in sorted_idx[1:]:
            if all(abs(idx - k) >= guard for k in kept):
                kept.append(idx)

        return [angles[k] for k in kept]

    def _find_angle_peaks(self, spectrum, angles, params):
        """
        在角度响应谱上做多峰检测, 替代 argmax 单峰.  (保留作为 fallback)

        两层过滤:
          第1层 find_peaks: 低阈值噪声门, 只过滤纯噪声波动
          第2层 手动dB检查: 峰顶/鞍部 vs prominence_db, 与噪底无关
        """
        if not params.get('aoa_multi_peak_en', False) or len(spectrum) < 3:
            return [angles[np.argmax(spectrum)]]

        # 噪底估计 (低百分位数, 仅用于噪声门)
        pct = params.get('aoa_noise_pct', 20)
        noise_floor = np.percentile(spectrum, pct)
        if noise_floor < 1e-12:
            noise_floor = 1e-12

        prominence_db = params.get('aoa_peak_prominence_db', 8.0)
        min_sep_deg = params.get('aoa_peak_min_sep_deg', 15.0)
        angle_span_deg = np.rad2deg(angles[-1] - angles[0])
        n_bins = len(angles)
        distance = max(1, int(min_sep_deg / angle_span_deg * n_bins))
        max_count = params.get('aoa_peak_max_count', 3)

        # === 第1层: 噪声门 ===
        # 用低的绝对 prominence, 只过滤纯噪声波动 (~5dB above noise)
        prominence_gate = noise_floor * 3.0
        peaks, props = signal.find_peaks(
            spectrum,
            height=noise_floor * 2.0,
            prominence=prominence_gate,
            distance=distance,
        )

        if len(peaks) == 0:
            return [angles[np.argmax(spectrum)]]

        # === 第2层: 真实 dB 比值检查 ===
        # prominence_db 衡量: 峰顶功率 ÷ 连接更高峰的鞍部功率
        # 这个比值只取决于峰和鞍部的相对关系, 与全局噪底无关
        sorted_idx = np.argsort(spectrum[peaks])[::-1]
        kept = [peaks[sorted_idx[0]]]  # 最强峰无条件保留

        for pi in sorted_idx[1:]:
            p = peaks[pi]
            pwr_peak = spectrum[p]

            # 找到连接此峰到任意已保留更高峰的鞍部 (区间最低点)
            saddle = pwr_peak
            for hp in kept:
                lo, hi = (p, hp) if p < hp else (hp, p)
                if hi - lo >= 2:
                    saddle = min(saddle, np.min(spectrum[lo:hi+1]))

            if saddle > 1e-12:
                ratio_db = 10.0 * np.log10(pwr_peak / saddle)
            else:
                ratio_db = 99.0

            if ratio_db >= prominence_db:
                kept.append(p)

        kept = kept[:max_count]
        return [angles[p] for p in kept]

    def _fft_angle_response(self, phase_vec, params, angles_rad):
        """FFT-based 角度响应: Bartlett BF in fine angular grid"""
        return self._bartlett_spectrum(phase_vec, angles_rad)

    def _dbf_angle_response(self, phase_vec, angles_rad):
        """DBF 角度响应: 使用预计算的引导矢量"""
        if getattr(self, '_dbf_sv_cache', None) is None:
            return self._bartlett_spectrum(phase_vec, angles_rad)

        # 寻找 angles_rad 中最接近预计算角度的索引
        x = phase_vec.reshape(-1, 1)
        pwr = np.abs(self._dbf_sv_cache.conj().T @ x) ** 2
        pwr = pwr.flatten()

        # 插值到请求的角度
        response = np.interp(angles_rad, self._dbf_angles_cache, pwr,
                             left=pwr[0], right=pwr[-1])
        return response

    def _music_angle_response(self, phase_vec, params, angles_rad):
        """MUSIC 角度响应: 伪谱值"""
        N = len(phase_vec)
        d = params.get('antenna_spacing', 0.5)
        # 构建协方差矩阵 (单快照 → 前后向平滑)
        x = phase_vec.reshape(-1, 1)
        R = x @ x.conj().T
        if params.get('music_forward_backward', False):
            J = np.flip(np.eye(N), axis=0)
            R = 0.5 * (R + J @ R.conj() @ J)

        # 特征分解
        evals, evecs = np.linalg.eigh(R)
        noise_subspace = evecs[:, :-1]  # 假设单目标

        n_indices = np.arange(N)
        pseudo = np.zeros(len(angles_rad), dtype=float)
        for i, theta in enumerate(angles_rad):
            sv = np.exp(1j * 2 * np.pi * d * np.sin(theta) * n_indices)
            v = sv.reshape(-1, 1)
            denom = v.conj().T @ noise_subspace @ noise_subspace.conj().T @ v
            pseudo[i] = 1.0 / (np.abs(denom[0, 0]) + 1e-12)
        return pseudo

    def _capon_angle_response(self, phase_vec, params, angles_rad):
        """Capon (MVDR) 角度响应谱, 用于 multipass CFAR 的 Pass 2/3.
        使用真实天线位置构建导向矢量, 支持 ant_calib_en / ant_calib_phase 相位校准.
        校准方式与 DBF 的 _init_dbf_steering 对齐: 校准量作用于引导矢量."""
        N = len(phase_vec)
        diag_load = params.get("capon_diag_load", 1e-3)

        # --- 预计算引导矢量的相位校准因子 (对齐 DBF / Dubhe Section 3.8.3) ---
        ant_x, channels = self._get_capon_antenna_x(params)
        if params.get('ant_calib_en', False):
            calib_phase = np.array(params.get('ant_calib_phase',
                                              [0.0] * 8), dtype=np.float64)
            calib_sv = np.exp(-1j * calib_phase[channels]).reshape(-1, 1)
        else:
            calib_sv = np.ones((N, 1), dtype=np.complex64)

        x = phase_vec.reshape(-1, 1)
        R = x @ x.conj().T

        if params.get('music_forward_backward', False):
            J = np.flip(np.eye(N), axis=0)
            R = 0.5 * (R + J @ R.conj() @ J)

        R += np.eye(N) * diag_load * np.abs(np.trace(R))

        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            return np.zeros(len(angles_rad))

        # 角度扫描 — 使用真实天线位置 + 引导矢量相位校准
        response = np.zeros(len(angles_rad))
        for i, theta in enumerate(angles_rad):
            sv = np.exp(1j * 2 * np.pi * ant_x * np.sin(theta)).reshape(-1, 1)
            sv = sv * calib_sv  # 相位校准作用于引导矢量 (与 DBF 一致)
            response[i] = 1.0 / np.real(sv.conj().T @ R_inv @ sv)

        return response

    def step_point_cloud_paper(self):
        """
        论文 Multipass CFAR 点云算法 (IEEE Sensors Journal 2024),
        适配 IR-UWB: NVE 底噪归一化 + min(L,R) 1D CFAR + Zoom-in

        Pass 0: NVE → SNR_map (线性), 消除 range 维度底噪起伏
        Pass 1: 1D Range CFAR — min(L,R) 噪声估计, 沿 Range 维度
        Pass 2: 1D Angle CFAR — min(L,R) 噪声估计, 沿 Angle 维度
        Pass 3: Zoom-in 角度细化 — 动态对比度阈值

        AoA 方法: MUSIC / DBF / FFT
        俯仰角: 不考虑
        """
        # 1. 获取数据
        all_c = self.dm.get_all_snapshot_as_array('complex')
        all_a = self.dm.get_all_snapshot_as_array('abs')
        if all_c is None:
            return None

        params = self.config.algo_params['POINT-CLOUD-PAPER']
        N_snaps = params['snapshots']
        cir_offset = params.get('cir_offset', 8)
        leakage_offset = params.get('leakage_offset', 5)

        # 跳过天线耦合区 + 取最近 N 帧
        current_cube = all_c[:, :, cir_offset:, -N_snaps:]

        # 2. Leakage roll + DC removal + window
        current_cube = np.roll(current_cube, -leakage_offset, axis=2)
        if params.get('doppler_dc_remove', True):
            current_cube = current_cube - np.mean(current_cube, axis=3, keepdims=True)
        if params.get('doppler_window') == 'chebyshev':
            atten = params.get('doppler_win_atten', 60)
            win = chebwin(current_cube.shape[3], at=atten)
            current_cube = current_cube * win[np.newaxis, np.newaxis, np.newaxis, :]

        # 3. Doppler FFT + 展平 + 选通道 + 非相干合并
        rd_cube = np.fft.fft(current_cube, axis=3)
        rd_flat = rd_cube.transpose(1, 0, 2, 3).reshape(
            8, rd_cube.shape[2], rd_cube.shape[3])
        valid_indices = params.get('indices_azimuth', [2, 3, 6, 7])
        if params.get('cfar_only_selected', True):
            rd_sel = rd_flat[valid_indices, :, :]
        else:
            rd_sel = rd_flat
        power_map = np.sum(np.abs(rd_sel) ** 2, axis=0)  # (Range, Doppler)
        n_range, n_dop = power_map.shape

        # 4. 速度门限掩码
        v_min = params.get('doppler_sum_v_min', 1)
        v_max = params.get('doppler_sum_v_max', 20)
        vel_mask = np.zeros(n_dop, dtype=bool)
        vel_mask[v_min:v_max] = True
        vel_mask[n_dop - v_max:n_dop - v_min] = True
        n_vel_bins = np.sum(vel_mask)

        # ==================== Pass 0: NVE 底噪归一化 ====================
        noise_floor_per_range = np.median(power_map, axis=1, keepdims=True)
        noise_floor_per_range = np.maximum(noise_floor_per_range, 1e-12)
        snr_map_linear = power_map / noise_floor_per_range  # (Range, Doppler)

        # ==================== Pass 1: Per-Doppler 1D Range CFAR ====================
        # 关键: 对每个 Doppler bin 独立做 1D Range CFAR, 而不是先 max 再 CFAR.
        # max() over N 个指数噪声的期望 ≈ H_N × μ ≈ 6× baseline (7.8 dB),
        # 使得本可检出的 14dB 目标被埋没.
        # per-bin 的 SNR 基线是 ~1 (0 dB), CFAR 可以直接在 SNR 域正常工作.
        rd_detections = []  # [(r_idx, d_idx, snr_linear), ...]
        for d in range(n_dop):
            if not vel_mask[d]:
                continue
            r_mask, _ = self.perform_1d_cfar_min(
                snr_map_linear[:, d],
                train=params['range_train'],
                guard=params['range_guard'],
                threshold_db=params['range_threshold_db'],
                ave_pad=params.get('range_ave_pad', 3)
            )
            for r_idx in np.argwhere(r_mask).flatten():
                rd_detections.append((int(r_idx), d, float(snr_map_linear[r_idx, d])))

        if len(rd_detections) == 0:
            peak_snr = 10 * np.log10(np.max(snr_map_linear[:, vel_mask]) + 1e-12)
            print(f"[POINT-CLOUD-PAPER] Pass1: 0 detections "
                  f"(peak SNR={peak_snr:.1f} dB, "
                  f"threshold={params['range_threshold_db']:.1f} dB, "
                  f"N_vel_bins={n_vel_bins})")
            return {
                "detected_points": [],
                "params": params,
                "breath_val": 0.0,
            }

        # 按 range 汇总: 每个 range bin 保留最强 SNR 的 Doppler bin
        best_per_range = {}  # r_idx → (d_idx, snr_linear)
        for r_idx, d_idx, snr_lin in rd_detections:
            if r_idx not in best_per_range or snr_lin > best_per_range[r_idx][1]:
                best_per_range[r_idx] = (d_idx, snr_lin)
        detected_ranges = sorted(best_per_range.keys())

        # ==================== 准备角度扫描 ====================
        coarse_n = params.get('aoa_coarse_n', 36)
        angle_deg_range = params.get('aoa_angle_range', [-70, 70])
        coarse_angles_deg = np.linspace(angle_deg_range[0], angle_deg_range[1], coarse_n)
        coarse_angles_rad = np.deg2rad(coarse_angles_deg)

        aoa_method = params.get('aoa_method', 'MUSIC')
        if aoa_method == 'DBF':
            channels = params.get('ant_dbf_select', valid_indices)
            # 指纹缓存：参数变化时自动重建（修复原实现改参数后不重建的隐患）
            fp = (tuple(channels), tuple(angle_deg_range), params.get('azimuth_num', 64),
                  params.get('ant_calib_en', False),
                  tuple(params.get('ant_calib_phase', [0.0] * 8)), params['center_freq'])
            if getattr(self, '_dbf_sv_cache', None) is None or getattr(self, '_dbf_cache_fp', None) != fp:
                # paper 路径天线模型（米制坐标）：与 optimized/dubhe 不同，勿合并
                all_virt_x = np.array([-0.038, 0.0, -0.038, -0.019,
                                       0.0, 0.038, 0.0, 0.019])
                self._dbf_angles_cache = np.linspace(
                    np.deg2rad(angle_deg_range[0]),
                    np.deg2rad(angle_deg_range[1]),
                    params.get('azimuth_num', 64))
                self._dbf_sv_cache = self._build_dbf_steering(
                    all_virt_x, channels, self._dbf_angles_cache,
                    params.get('ant_calib_en', False), params.get('ant_calib_phase', [0.0] * 8),
                    params['center_freq'])
                self._dbf_cache_fp = fp

        # ==================== Pass 2 & 3: Per detected Range ====================
        detected_points = []
        angle_quality_threshold = params.get('angle_threshold_db', 8.0)

        for r_idx in detected_ranges:
            d_idx, peak_snr_linear = best_per_range[r_idx]
            phase_vec_full = rd_flat[:, r_idx, d_idx]
            phase_vec_sel = phase_vec_full[valid_indices]
            peak_snr_db = 10 * np.log10(peak_snr_linear + 1e-12)

            # ============ Pass 2: 粗角度峰值检测 (替代 1D Angle CFAR) ============
            # 计算粗角度响应谱, 直接取峰值和角度质量, 不做 CFAR
            angle_response = self._compute_angle_response(
                phase_vec_sel, params, coarse_angles_rad)

            # 角度质量 = 峰值 / 中值 (类似 SNR)
            angle_median = np.median(angle_response)
            if angle_median < 1e-12:
                continue
            angle_peak_idx = np.argmax(angle_response)
            angle_peak_value = angle_response[angle_peak_idx]
            angle_snr_linear = angle_peak_value / angle_median
            angle_snr_db = 10 * np.log10(angle_snr_linear + 1e-12)

            # 角度质量不够 → 跳过 (相位向量可能来自噪声而非真实目标)
            if angle_snr_db < angle_quality_threshold:
                continue

            coarse_angle = coarse_angles_rad[angle_peak_idx]

            # ============ Pass 3: Zoom-in 细化 + 动态对比度阈值 ============
            zoom_factor = params.get('zoom_in_factor', 3)
            gamma = params.get('zoom_threshold_gamma', 0.5)
            coarse_step_deg = (angle_deg_range[1] - angle_deg_range[0]) / max(coarse_n - 1, 1)
            zoom_step_deg = coarse_step_deg / zoom_factor
            half_span = coarse_step_deg * 1.2

            zoom_ang_deg = np.arange(
                max(angle_deg_range[0], np.rad2deg(coarse_angle) - half_span),
                min(angle_deg_range[1], np.rad2deg(coarse_angle) + half_span + zoom_step_deg * 0.5),
                zoom_step_deg)
            if len(zoom_ang_deg) < 3:
                zoom_ang_deg = np.array([np.rad2deg(coarse_angle)])
            zoom_angles_rad = np.deg2rad(zoom_ang_deg)

            zoom_response = self._compute_angle_response(
                phase_vec_sel, params, zoom_angles_rad)

            G_max = np.max(zoom_response)
            G_min = np.min(zoom_response)
            if G_max + G_min < 1e-12:
                continue

            # 论文公式: 动态对比度阈值
            # P_th = P_peak × (γ - (Gmax-Gmin)/(Gmax+Gmin))
            contrast = (G_max - G_min) / (G_max + G_min)
            gamma_th = G_max * max(0.15, gamma - contrast)

            # 检出所有高于动态阈值的 zoom-in 角度
            best_angle = None
            best_zoom_snr = -np.inf
            for u, theta in enumerate(zoom_angles_rad):
                if zoom_response[u] > gamma_th:
                    zoom_snr = 10 * np.log10(zoom_response[u] / (G_min + 1e-12))
                    if zoom_snr > best_zoom_snr:
                        best_zoom_snr = zoom_snr
                        best_angle = theta

            # fallback: 取 zoom 响应最强点
            if best_angle is None:
                best_u = np.argmax(zoom_response)
                best_angle = zoom_angles_rad[best_u]
                best_zoom_snr = 10 * np.log10(zoom_response[best_u] / (G_min + 1e-12))

            # 校准
            calib_mode = params.get('aoa_calib_mode', 'linear')
            if calib_mode in ('linear', 'both'):
                offset_deg = params.get('aoa_offset_deg', 0.0)
                scale = params.get('aoa_scale', 1.0)
                best_angle = (best_angle * scale) + np.deg2rad(offset_deg)
            best_angle = np.clip(best_angle, -np.pi / 2, np.pi / 2)

            # 坐标转换
            dist = r_idx * params['dist_per_tap']
            point_x = dist * np.sin(best_angle)
            point_y = -dist * np.cos(best_angle)

            # SNR: 使用 RD 域的 SNR (与 DUBHE 模式一致, 有物理意义)
            detected_points.append({
                'pos': (point_x, point_y),
                'snr': float(peak_snr_db),
                'time': time.time()
            })

        # debug 统计
        n_points = len(detected_points)
        if n_points > 0 or len(detected_ranges) > 0:
            print(f"[POINT-CLOUD-PAPER] Pass1: {len(detected_ranges)}/{n_range} ranges → "
                  f"Pass3: {n_points} points "
                  f"(SNR range: {np.min([p['snr'] for p in detected_points]) if n_points else 0:.1f}-"
                  f"{np.max([p['snr'] for p in detected_points]) if n_points else 0:.1f} dB)")

        # ==================== Breathing ====================
        dop_fft_n = params['snapshots']
        val_breath = 0.0
        try:
            if all_a.shape[3] >= dop_fft_n:
                dop_tx = params['doppler_tx']
                dop_rx = params['doppler_rx']
                tx_idx = self.config.udp_tx_list.index(dop_tx)
                rx_idx = self.config.udp_rx_list.index(dop_rx)
                recent_a = all_a[rx_idx, tx_idx, :, -dop_fft_n:]
                val_breath, _, _, max_motion = find_breathing_feature(
                    np.fft.fft(recent_a, axis=1),
                    self.config.snapshot_rate, 0.15, 0.7, 0, 16)
                if self.breathing_window.maxlen != 10:
                    self.breathing_window = deque(list(self.breathing_window), maxlen=10)
                self.breathing_window.append(val_breath)
        except:
            pass
        val_breath = float(val_breath)

        return {
            "detected_points": detected_points,
            "params": params,
            "breath_val": val_breath,
        }

    # ===== DBF & CFAR helpers (shared by OPTIMIZED and DUBHE) =====

    def _subbin_refine(self, power_map, r_idx, d_idx):
        """二次插值细化，返回 (r_fine, d_fine)"""
        R, D = power_map.shape
        r_lo, r_hi = max(0, r_idx - 1), min(R - 1, r_idx + 1)
        d_lo, d_hi = max(0, d_idx - 1), min(D - 1, d_idx + 1)

        r_fine = float(r_idx)
        if r_lo < r_idx < r_hi:
            a, b, c = power_map[r_lo, d_idx], power_map[r_idx, d_idx], power_map[r_hi, d_idx]
            denom = a - 2 * b + c
            if abs(denom) > 1e-12:
                r_fine = r_idx + (a - c) / (2 * denom)

        d_fine = float(d_idx)
        if d_lo < d_idx < d_hi:
            a, b, c = power_map[r_idx, d_lo], power_map[r_idx, d_idx], power_map[r_idx, d_hi]
            denom = a - 2 * b + c
            if abs(denom) > 1e-12:
                d_fine = d_idx + (a - c) / (2 * denom)

        return r_fine, d_fine

    def _init_dbf_steering(self, params):
        """预计算 DBF 引导矢量矩阵（OPTIMIZED 和 DUBHE 共用）"""
        wavelength = 2.99792458e8 / params['center_freq']
        channels = params.get('ant_dbf_select', params.get('siso_ch', [2, 3, 6, 7]))
        # 8通道虚拟天线 x 坐标 (m): [TX1-RX4, TX1-RX5, TX1-RX6, TX1-RX7,
        #                            TX2-RX4, TX2-RX5, TX2-RX6, TX2-RX7]
        all_virt_x = np.array([0.0 * wavelength, 0.5 * wavelength, 1 * wavelength, 0.5 * wavelength,
                               0.5 * wavelength, wavelength, 0 * wavelength, -0.5 * wavelength])
        azi_deg = np.linspace(params['azi_angle_range'][0],
                              params['azi_angle_range'][1],
                              params['azimuth_num'])
        self._dbf_angles = np.deg2rad(azi_deg)
        self._dbf_sv = self._build_dbf_steering(
            all_virt_x, channels, self._dbf_angles,
            params.get('ant_calib_en', False), params.get('ant_calib_phase', [0] * 8),
            params['center_freq'])

    @staticmethod
    def _build_dbf_steering(all_virt_x, channels, angles_rad, calib_en, calib_phase, center_freq):
        """构造 DBF 引导矢量矩阵（两种天线模型共用公式，阶段 3 统一）。

        all_virt_x: 8 通道虚拟天线 x 坐标（米或波长倍数，统一除以波长归一）；
        angles_rad: 调用方构造好的角度网格（保证浮点行为与原来完全一致）。
        """
        wavelength = 2.99792458e8 / center_freq
        ant_x = np.asarray(all_virt_x, dtype=np.float64)[channels] / wavelength
        sv = np.exp(-1j * 2 * np.pi * ant_x[:, np.newaxis] * np.sin(angles_rad[np.newaxis, :]))
        if calib_en:
            calib = np.exp(1j * np.asarray(calib_phase, dtype=np.float64)[channels])
            sv = sv * calib[:, np.newaxis]
        return sv

    def _init_capon_steering(self, params):
        """预计算 Capon 导向矢量矩阵, 与 calculate_capon_aoa 完全对齐"""
        ant_x, channels = self._get_capon_antenna_x(params)
        # 相位校准因子 (与 calculate_capon_aoa 一致)
        if params.get('ant_calib_en', False):
            calib_phase = np.array(params.get('ant_calib_phase',
                                  [0.0] * 8), dtype=np.float64)
            calib_sv = np.exp(1j * calib_phase[channels]).reshape(-1, 1)
        else:
            calib_sv = np.ones((len(channels), 1), dtype=np.complex64)

        azi_deg = np.linspace(params['azi_angle_range'][0],
                              params['azi_angle_range'][1],
                              params['azimuth_num'])
        self._capon_angles = np.deg2rad(azi_deg)
        # sv = exp(1j * 2π * ant_x * sin(θ))  ← 与 calculate_capon_aoa 完全一致
        self._capon_sv = np.exp(1j * 2 * np.pi *
                                ant_x[:, np.newaxis] * np.sin(self._capon_angles[np.newaxis, :]))
        # sv = sv * calib_sv  ← 与 calculate_capon_aoa 一致
        self._capon_sv = self._capon_sv * calib_sv

    def _compute_ra_heatmap(self, params):
        """
        计算 Range-Azimuth 热力图 (Bartlett/DBF, 多chirp协方差).

        与论文 Section III 对齐:
          - 不做 Doppler FFT
          - 每个 range bin 用全部 chirp 构造协方差矩阵 R_k (P×P)
          - Bartlett 波束形成: H(k,θ) = a(θ)^H · R_k · a(θ)
          → 等价于每 chirp 分别 DBF 后跨 chirp 平均功率
          → SNR 增益 ~10log10(L) vs 单 chirp

        返回:
          H:         (K, I) 功率热力图 (线性值)
          ranges_m:  (K,) 距离数组 (米)
          angles_deg:(I,) 角度数组 (度)
        """
        all_c = self.dm.get_all_snapshot_as_array('complex')
        if all_c is None:
            return None, None, None

        N_snaps = params['snapshots']
        cir_comb = params.get('cir_combine_num', 1)
        leakage_offset = params['leakage_offset']
        current_cube = all_c[:, :, :, -N_snaps:]

        # 预处理
        if cir_comb > 1:
            n_comb = current_cube.shape[3] // cir_comb
            trim = n_comb * cir_comb
            current_cube = current_cube[:, :, :32, :trim] \
                .reshape(4, 2, 32, n_comb, cir_comb).mean(axis=4)
        current_cube, range_bin_start = apply_range_bin_selection(
            current_cube, params, return_start_bin=True)

        # 展平 8 通道 → 选 4 通道
        # current_cube: (4 RX, 2 TX, 32 range, L chirps)
        cube_flat = current_cube.transpose(1, 0, 2, 3).reshape(
            8, current_cube.shape[2], current_cube.shape[3])
        valid_indices = params.get('indices_azimuth', [2, 3, 6, 7])
        A_all = cube_flat[valid_indices, :, :]  # (4, 32, L)

        # 初始化 Capon 导向矢量 (用真实天线位置, 非 ULA 假设)
        if getattr(self, '_capon_sv', None) is None:
            self._init_capon_steering(params)
        I = len(self._capon_angles)
        K = A_all.shape[1]
        L = A_all.shape[2]

        # 对每个 range bin: Capon 波束形成 (论文 Eq.11-13)
        # R_k = A_k·A_k^H / L + βI,  H(k,θ) = 1 / Re(a(θ)^H · R_k^{-1} · a(θ))
        diag_load = params.get('capon_diag_load', 1e-3)
        H = np.zeros((K, I), dtype=np.float64)
        R_inv_list = []
        for k_idx in range(K):
            A_k = A_all[:, k_idx, :]
            R_k = (A_k @ A_k.conj().T) / L
            R_k += np.eye(4) * diag_load * np.abs(np.trace(R_k))
            try:
                R_inv = np.linalg.inv(R_k)
            except np.linalg.LinAlgError:
                R_inv_list.append(None)
                continue
            R_inv_list.append(R_inv)
            denom = np.sum((self._capon_sv.conj() * (R_inv @ self._capon_sv)), axis=0)
            H[k_idx, :] = 1.0 / np.real(np.clip(denom, 1e-12, None))

        range_zero_bin = float(params.get('range_zero_bin', range_bin_start))
        range_start_bins = max(0.0, range_bin_start - range_zero_bin)
        ranges_m = (range_start_bins + np.arange(K)) * params['dist_per_tap']
        angles_deg = np.rad2deg(self._capon_angles)
        return H, ranges_m, angles_deg, A_all, R_inv_list

    def _two_pass_cfar(self, H, params):
        """
        Algorithm 1: Two-Pass CFAR (论文 IEEE JSEN 2024)

        Pass 1 — Range-wise: 沿 range 维度对每个 azimuth bin 做 1D CFAR.
        Pass 2 — Azimuth-wise: 只对 Pass1 有检出的 range bin, 沿 azimuth 维做 1D CFAR.

        参数 (来自 params):
          pass1_N_W_R, pass1_N_G_R, pass1_N_ave_R, pass1_gamma_R_db
          pass2_N_W_phi, pass2_N_G_phi, pass2_gamma_phi_db

        返回:
          detections: List[dict], 每个检出包含 {range_idx, azimuth_idx, range_m, angle_deg, power}
        """
        K, I = H.shape  # K=range bins, I=azimuth bins
        debug = params.get('aoa_debug_print', False)

        # ─── Pass 1: Range-wise CFAR ───
        N_W_R = params.get('pass1_N_W_R', 3)
        N_G_R = params.get('pass1_N_G_R', 1)
        N_ave_R = params.get('pass1_N_ave_R', 2)
        gamma_R_db = params.get('pass1_gamma_R_db', 10.0)
        gamma_R = 10 ** (gamma_R_db / 10.0)
        N_WG_R = N_W_R + N_G_R

        C = np.zeros((K, I), dtype=bool)

        for i in range(I):
            # 边界 padding: 取首尾 N_ave_R 个 range bin 的均值
            mu_L = np.mean(H[:N_ave_R, i]) if N_ave_R > 0 else H[0, i]
            mu_R = np.mean(H[K - N_ave_R:, i]) if N_ave_R > 0 else H[-1, i]

            # 构造 padded 向量: [mu_L (×N_WG_R), H[:,i], mu_R (×N_WG_R)]
            H_pad = np.concatenate([
                np.full(N_WG_R, mu_L),
                H[:, i],
                np.full(N_WG_R, mu_R)
            ])
            K_prime = len(H_pad)

            for k in range(K):
                # μ_L = 左侧训练单元均值 (在 H_pad 中的索引)
                lo_L = k
                hi_L = k + N_W_R
                # μ_R = 右侧训练单元均值
                lo_R = k + 2 * N_G_R + N_W_R + 1
                hi_R = k + 2 * N_G_R + 2 * N_W_R + 1
                if hi_R > K_prime:
                    hi_R = K_prime
                    lo_R = max(0, hi_R - N_W_R)

                mu_L_val = np.mean(H_pad[lo_L:hi_L]) if hi_L > lo_L else 0.0
                mu_R_val = np.mean(H_pad[lo_R:hi_R]) if hi_R > lo_R else np.inf
                W_R = min(mu_L_val, mu_R_val)

                if W_R > 1e-12 and H[k, i] > gamma_R * W_R:
                    C[k, i] = True

        if debug:
            n_pass1 = np.sum(C)
            print(f"[RA-CFAR] Pass1 (Range): {n_pass1} detections across {K}×{I} cells")

        # ─── Pass 2: Azimuth-wise CFAR ───
        N_W_phi = params.get('pass2_N_W_phi', 4)
        N_G_phi = params.get('pass2_N_G_phi', 2)
        gamma_phi_db = params.get('pass2_gamma_phi_db', 10.0)
        gamma_phi = 10 ** (gamma_phi_db / 10.0)
        N_WG_phi = N_W_phi + N_G_phi

        for k in range(K):
            # 只对有 Pass1 检出的 range bin 做 Pass2
            if not np.any(C[k, :]):
                continue

            # 循环 padding: 首尾各取 N_WG_phi 个 azimuth bin 环绕
            # H_pad = [H[k, I-N_WG_phi : I-1], H[k, 0 : I-1], H[k, 0 : N_WG_phi-1]]
            # 对应伪代码: 末尾取最后一个 elevation 角度的, 前面取第一个 elevation 的
            # 我们 M=1, 简化为自身的循环 padding
            pad_left = H[k, I - N_WG_phi:I]   # 末尾 N_WG_phi 个
            pad_right = H[k, :N_WG_phi]       # 开头 N_WG_phi 个
            H_pad_phi = np.concatenate([pad_left, H[k, :], pad_right])
            I_prime = len(H_pad_phi)

            for i in range(I):
                # 在 H_pad_phi 中, 原始 H[k,i] 对应索引 i + N_WG_phi
                # μ_L = 左侧训练单元
                lo_L = i
                hi_L = i + N_W_phi
                # μ_R = 右侧训练单元
                lo_R = i + 2 * N_G_phi + N_W_phi + 1
                hi_R = i + 2 * N_G_phi + 2 * N_W_phi + 1
                if hi_R > I_prime:
                    hi_R = I_prime
                    lo_R = max(0, hi_R - N_W_phi)

                mu_L_val = np.mean(H_pad_phi[lo_L:hi_L]) if hi_L > lo_L else 0.0
                mu_R_val = np.mean(H_pad_phi[lo_R:hi_R]) if hi_R > lo_R else np.inf
                W_phi = min(mu_L_val, mu_R_val)

                if W_phi > 1e-12 and H[k, i] > gamma_phi * W_phi:
                    # 已经通过 Pass1, 再通过 Pass2 → 确认检出
                    pass
                else:
                    C[k, i] = False  # Pass2 未通过 → 撤销

        if debug:
            n_pass2 = np.sum(C)
            print(f"[RA-CFAR] Pass2 (Azimuth): {n_pass2} final detections")

        # ─── 组装检出结果 ───
        ranges_m = np.arange(K) * params['dist_per_tap']
        angles_deg = np.rad2deg(self._capon_angles)

        detections = []
        hit_indices = np.argwhere(C)
        for k, i in hit_indices:
            dist = ranges_m[k]
            az_deg = angles_deg[i]
            # 校准
            calib_mode = params.get('aoa_calib_mode', 'linear')
            if calib_mode in ('linear', 'both'):
                az_deg = az_deg * params.get('aoa_scale', 1.0) + params.get('aoa_offset_deg', 0.0)
            az_rad = np.deg2rad(np.clip(az_deg, -90, 90))
            point_x = dist * np.sin(az_rad)
            point_y = -dist * np.cos(az_rad)
            detections.append({
                'pos': (-point_x, point_y),
                'snr': 10 * np.log10(H[k, i] + 1e-12),
                'time': time.time(),
                'range_idx': int(k),
                'azimuth_idx': int(i),
            })

        return detections

    def _dbf_estimate(self, phase_vec, params):
        """DBF 方位角估计，返回角度列表 (弧度) — 支持多峰检测"""
        debug = params.get('aoa_debug_print', False)
        x = phase_vec.reshape(-1, 1)
        pwr = np.abs(self._dbf_sv.conj().T @ x) ** 2
        pwr = pwr.flatten()

        if debug:
            # --- 诊断打印: 相位向量 ---
            ch_mag = np.abs(phase_vec)
            ch_phase = np.angle(phase_vec, deg=True)
            print(f"\n{'='*60}")
            print(f"[DBF DEBUG] 4通道相位向量:")
            for i in range(len(phase_vec)):
                print(f"  ch[{i}]: mag={ch_mag[i]:.4f} ({20*np.log10(ch_mag[i]+1e-12):.1f} dB), "
                      f"phase={ch_phase[i]:.1f}°")
            # --- 诊断打印: 角度谱概览 ---
            med = np.median(pwr)
            pct20 = np.percentile(pwr, 20)
            top_idx = np.argmax(pwr)
            print(f"[DBF DEBUG] 角度谱概览 (64 bins, -70°~70°):")
            print(f"  最强峰: θ={np.rad2deg(self._dbf_angles[top_idx]):.1f}°, "
                  f"power={pwr[top_idx]:.2f} ({10*np.log10(pwr[top_idx]+1e-12):.1f} dB linear)")
            print(f"  noise_floor: median={med:.2f}, 20%ile={pct20:.2f}")
            # 列出所有局部峰 (手动扫一遍, 不依赖 find_peaks)
            all_local_peaks = []
            for j in range(2, len(pwr)-2):
                if pwr[j] > pwr[j-1] and pwr[j] > pwr[j-2] and pwr[j] > pwr[j+1] and pwr[j] > pwr[j+2]:
                    all_local_peaks.append((np.rad2deg(self._dbf_angles[j]), pwr[j]))
            all_local_peaks.sort(key=lambda v: -v[1])
            print(f"  所有局部峰 (前5个):")
            for deg, val in all_local_peaks[:5]:
                print(f"    θ={deg:+.1f}°, power={val:.2f} ({10*np.log10(val+1e-12):.1f} dB)")

        # 多峰检测: 优先走角度CFAR, 否则 fallback 到 prominence 寻峰
        if params.get('aoa_cfar_en', False):
            az_list = self._angle_cfar_detect(pwr, self._dbf_angles, params)
        else:
            az_list = self._find_angle_peaks(pwr, self._dbf_angles, params)
        n_azi = len(self._dbf_angles)

        # 主瓣滤波 (dbf_diff): 对每个峰独立检查
        dbf_diff = params.get('dbf_diff', 0)
        if dbf_diff > 0:
            filtered = []
            for az in az_list:
                peak_idx = np.argmin(np.abs(self._dbf_angles - az))
                guard = max(3, n_azi // 16)
                side_mask = np.ones(n_azi, dtype=bool)
                lo = max(0, peak_idx - guard)
                hi = min(n_azi, peak_idx + guard + 1)
                side_mask[lo:hi] = False
                side_pwr = np.mean(pwr[side_mask]) if np.any(side_mask) else 0.0
                if side_pwr > 1e-12:
                    ratio_db = 10 * np.log10(pwr[peak_idx] / side_pwr)
                    if ratio_db >= dbf_diff:
                        filtered.append(az)
                else:
                    filtered.append(az)
            az_list = filtered

        if debug:
            print(f"[DBF DEBUG] _find_angle_peaks 返回 (dbf_diff前): "
                  f"{[f'{np.rad2deg(a):.1f}°' for a in az_list]}")
            # 打印 dbf_diff 过滤细节
            if dbf_diff > 0:
                print(f"[DBF DEBUG] dbf_diff={dbf_diff}dB 过滤后: "
                      f"{[f'{np.rad2deg(a):.1f}°' for a in az_list]}")

        if len(az_list) == 0:
            if debug:
                print(f"[DBF DEBUG] 所有峰被过滤! 返回 0.0°")
            return [0.0]

        # 校准: 对每个角度应用
        calib_mode = params.get('aoa_calib_mode', 'linear')
        offset_deg = params.get("aoa_offset_deg", 0.0)
        scale = params.get("aoa_scale", 1.0)
        offset_rad = np.deg2rad(offset_deg)

        result = []
        for az in az_list:
            if calib_mode in ('linear', 'both'):
                az = (az * scale) + offset_rad
            az = np.clip(az, -np.pi / 2, np.pi / 2)
            result.append(az)

        if debug:
            print(f"[DBF DEBUG] 最终输出 (校准后): "
                  f"{[f'{np.rad2deg(a):.1f}°' for a in result]}")
            print(f"{'='*60}\n")

        return result  # List[float]

    def step_point_cloud_dubhe(self):
        """
        Dubhe CPD 风格点云算法 (2D only):
        相干积累 → ring buffer → leakage roll → Chebyshev 窗 → DC 去除 → Doppler FFT
        → SISO 非相干合并 → NVE 噪底估计 → SNR 图 → CFAR 检测
        → 速度/距离门限 → 峰值滤波 → 5 区动态后滤波 → DBF 方位角
        """
        all_c = self.dm.get_all_snapshot_as_array('complex')
        all_a = self.dm.get_all_snapshot_as_array('abs')
        if all_c is None:
            return None

        params = self.config.algo_params['POINT-CLOUD-DUBHE']
        cir_comb = params['cir_combine_num']
        ring_len = params['ring_buffer_len']
        slide = params['slide_step']

        # --- Phase 1: 相干积累 (post background-removal) ---
        n_frames = all_c.shape[3]
        for i in range(n_frames):
            frame = all_c[:, :, :, i]  # (4, 2, 32)
            self.dubhe_accum += frame
            self.dubhe_accum_cnt += 1
            if self.dubhe_accum_cnt >= cir_comb:
                combined = self.dubhe_accum / cir_comb
                self.dubhe_ring.append(combined)
                self.dubhe_new_comb_cnt += 1
                self.dubhe_accum = np.zeros((4, 2, 32), dtype=np.complex64)
                self.dubhe_accum_cnt = 0

        # --- Phase 2: 判断是否输出 ---
        if len(self.dubhe_ring) < ring_len:
            return None
        if self.dubhe_new_comb_cnt < slide:
            return None
        self.dubhe_new_comb_cnt = 0

        # --- Phase 3: 取最近 ring_len 帧并转置 ---
        ring_data = np.array(list(self.dubhe_ring)[-ring_len:])  # (ring_len, 4, 2, 32)
        cube_2d = ring_data.transpose(1, 2, 3, 0)  # (4, 2, 32, ring_len)

        # --- Phase 4: leakage roll ---
        cube_2d = np.roll(cube_2d, -params['leakage_offset'], axis=2)

        # --- Phase 5: DC removal + Chebyshev window ---
        if params['doppler_dc_en']:
            cube_2d = cube_2d - np.mean(cube_2d, axis=3, keepdims=True)
        if params['doppler_win_en']:
            win = chebwin(cube_2d.shape[3], at=params['doppler_win_coef'])
            cube_2d = cube_2d * win[np.newaxis, np.newaxis, np.newaxis, :]

        # --- Phase 6: Doppler FFT ---
        n_fft = params['doppler_fft']
        rd_cube = np.fft.fft(cube_2d, n=n_fft, axis=3)

        # --- Phase 7: SISO 非相干合并 ---
        rd_flat = rd_cube.transpose(1, 0, 2, 3).reshape(8, rd_cube.shape[2], n_fft)
        siso_ch = params['siso_ch']
        rd_sel = rd_flat[siso_ch, :, :]
        power_map = np.sum(np.abs(rd_sel) ** 2, axis=0)

        # --- Phase 8: NVE (沿 Doppler 轴取中值) ---
        noise_floor = np.median(power_map, axis=1, keepdims=True)
        noise_floor = np.maximum(noise_floor, 1e-12)

        # --- Phase 9: SNR 图 + CFAR 初次筛选 ---
        snr_map = 10 * np.log10(power_map / noise_floor)
        cfar_th = params['cfar_th'][0]
        mask = snr_map > cfar_th

        # --- Phase 10: 距离/速度门 + 峰值滤波 ---
        low_r, high_r = params['cfar_low_r_idx'], params['cfar_high_r_idx']
        low_v, high_v = params['cfar_low_v_idx'], params['cfar_high_v_idx']
        mask[:low_r, :] = False
        mask[high_r:, :] = False

        vel_mask = np.zeros(n_fft, dtype=bool)
        vel_mask[low_v:high_v] = True
        vel_mask[n_fft - high_v:n_fft - low_v] = True
        mask &= vel_mask[np.newaxis, :]

        if params['cfar_doppler_peak_flag']:
            mask &= (power_map == ndimage.maximum_filter(power_map, size=(1, 3)))
        if params['cfar_range_peak_flag']:
            mask &= (power_map == ndimage.maximum_filter(power_map, size=(3, 1)))

        # --- Phase 11: 5 区动态后滤波 ---
        # 注意: noi_edges 的绝对值依赖硬件 ADC 标定。此处用 SNR 图的中值
        # 作为环境噪声代理指标，自动选择动态门限。
        if params.get('dynamic_filter_en', True):
            noise_level_val = np.median(noise_floor)
            noise_db = 10 * np.log10(noise_level_val + 1e-12)
            edges = params['noi_edges']
            dyn_th = params['cfar_th_dynamic']
            zone_idx = np.searchsorted(edges, noise_db)
            # clamp to valid range
            zone_idx = max(0, min(zone_idx, len(dyn_th) - 1))
            dyn_threshold = dyn_th[zone_idx]
            mask &= snr_map > dyn_threshold

        # --- Phase 12: 检出 + 子网格细化 + AoA ---
        aoa_method = params.get('aoa_method', 'DBF')
        if aoa_method == 'DBF' and getattr(self, '_dbf_sv', None) is None:
            self._init_dbf_steering(params)

        hit_indices = np.argwhere(mask)
        detected_points = []
        for r_idx, d_idx in hit_indices:
            r_fine, d_fine = self._subbin_refine(snr_map, r_idx, d_idx)

            phase_vec = rd_flat[siso_ch, r_idx, d_idx]
            if aoa_method == 'DBF':
                az_list = self._dbf_estimate(phase_vec, params)
            else:
                az_list = self.estimate_aoa(phase_vec, params)

            dist = r_fine * params['dist_per_tap']
            snr_val = snr_map[r_idx, d_idx]
            for az_angle in az_list:
                point_x = dist * np.sin(az_angle)
                point_y = -dist * np.cos(az_angle)
                detected_points.append({
                    'pos': (point_x, point_y),
                    'snr': snr_val,
                    'time': time.time()
                })

        # --- Phase 13: Breathing ---
        val_breath = 0.0
        try:
            if all_a.shape[3] >= 16:
                dop_tx = params['doppler_tx']; dop_rx = params['doppler_rx']
                tx_idx = self.config.udp_tx_list.index(dop_tx)
                rx_idx = self.config.udp_rx_list.index(dop_rx)
                recent_a = all_a[rx_idx, tx_idx, :, -16:]
                val_breath, _, _, max_motion = find_breathing_feature(
                    np.fft.fft(recent_a, axis=1),
                    self.config.snapshot_rate, 0.15, 0.7, 0, 16)
                if self.breathing_window.maxlen != 10:
                    self.breathing_window = deque(list(self.breathing_window), maxlen=10)
                self.breathing_window.append(val_breath)
        except:
            pass
        val_breath = float(val_breath)
        return {
            "detected_points": detected_points,
            "params": self.config.algo_params['POINT-CLOUD-DUBHE'],
            "breath_val": val_breath,
        }



# ==============================================================================
# 4.5 Seat Occupancy Detector (论文: Vehicle Occupancy Detector Based on
#     FMCW mm-Wave Radar at 77 GHz, Section IV)
