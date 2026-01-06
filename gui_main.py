import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import json
import threading
import queue
import serial
import os
import struct
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from scipy import ndimage, signal

# --- External Modules ---
from UDP_data_listen import UDPFrameServer
import util 
try:
    from breathe import find_breathing_feature
except ImportError:
    print("Warning: breathe.py not found. Breathing feature will be disabled.")
    find_breathing_feature = lambda *args: (0,0,0,0)

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==============================================================================
# 0. Presets & Defaults
# ==============================================================================
ANTENNA_LAYOUTS = {
    "2x4_Default": {
        "tx": [[-0.019, 0, 0.0], [0.019, 0, 0.0]],
        "rx": [[-0.019, 0, 0.0], [0.019, 0, 0.0], [-0.019, 0.0, -0.02], [0.0, 0.0, -0.02]],
        "tx_list": [1, 2], "rx_list": [4, 5, 6, 7],
        "tx_num": 2, "rx_num": 4, "rx_start": 4
    },
    "4x4_Demo": {
        "tx": [[-0.03,0,0], [-0.01,0,0], [0.01,0,0], [0.03,0,0]],
        "rx": [[-0.03,0,0], [-0.01,0,0], [0.01,0,0], [0.03,0,0]],
        "tx_list": [1, 2, 3, 4], "rx_list": [5, 6, 7, 8],
        "tx_num": 4, "rx_num": 4, "rx_start": 5
    }
}

DEFAULT_ALGO_PARAMS = {
    "PLOT": {
        "tx_pair": 1, 
        "rx_pair": 6, 
        "show_raw": True,
        "ylim_min": -100, 
        "ylim_max": 100
    },
    "2D-MUSIC": {
        # --- 核心物理参数 (2D-MUSIC 专属) ---
        "center_freq": 7.9872e9,  # CF
        "bandwidth": 100e6,       # BW
        "epsilon_r": 1.0,         # 介电常数
        "system_delay": 0.0,      # 系统延时补偿 (s)
        "cir_offset": 5,
        # --- 算法逻辑参数 ---
        "music_snapshots": 16,    # 参与计算的快照数
        "mdl_max_targets": 1,     # MDL 最大搜索目标数
        "forward_backward": False,# 前后向平滑 (解相干)
        "diag_load": 1e-5,        # 对角加载因子
        "fov_degrees": 180.0,     # 视场角
        
        # --- 绘图与检测参数 ---
        "music_threshold": 0.15,  # 伪谱显示阈值
        "music_vmax": 0.6,        # <--- 新增：手动设置伪谱显示上限 (对比度)
        "virt_tx_idx": [0, 1],    # 虚拟阵列 TX 索引
        "virt_rx_idx": [2, 3],    # 虚拟阵列 RX 索引
        "fusion_mode": "cartesian", # 'polar' 或 'cartesian'
        
        # --- 多普勒与呼吸参数 ---
        "doppler_tx": 1,
        "doppler_rx": 6,
        "doppler_fft": 64,
        "dop_th_start": 1200,
        "dop_th_default": 1000,
        "smooth_win": 10,
        "breath_val_th": 0.25,
        "breath_cnt_th": 5,
        
        # --- 绘图范围 (米) ---
        "plot_xlim": 2.0,
        "plot_ylim_min": -4.0,
        "plot_ylim_max": 0.0
    },
    "POINT-CLOUD":{
        "center_freq": 7.9872e9,
        "bandwidth": 100e6,
        "snapshots": 64,            # 论文规定：每帧包含 32 个 Chirp
        "cfar_threshold_db": 10.0,  # 论文规定：检测阈值系数为 15 dB
        "music_forward_backward": True,  # <--- 新增：默认为 False
        # 论文中 CA-CFAR 的窗口设置
        "range_train": 3,           
        "range_guard": 1,           
        "doppler_train": 4,         
        "doppler_guard": 2,         
        "point_lifetime_sec": 1.5,  # 点云持久化：1.5秒后消失
        "dist_per_tap": 0.1875,     # 距离分辨率参考 (c/2B)
        "plot_xlim": 1.5,
        "plot_ylim_min": -2.5,
        "plot_ylim_max": 0.0,
        "aoa_method": "MUSIC",        # 可选 "FFT" 或 "MUSIC"
        "fft_n": 16,                # AoA FFT 的补零点数，增加点数可提高角度分辨率
        "num_virtual_ant": 8,       # 论文中定义的 8 个等效虚拟天线
        "antenna_spacing": 0.5,     # 天线间距为 0.5 lambda
        "indices_azimuth" : [2, 3,6,7],
        "cfar_only_selected": True,  # <--- 新增：True 表示 CFAR 仅计算选中的天线，False 表示计算全部
        "cir_offset" : 8,
        "aoa_offset_deg": -12.0,    # 角度偏移（度），正值代表向右偏
        "aoa_scale": 1.0,         # 角度缩放比例，默认为 1.0
        "doppler_tx": 1,
        "doppler_rx": 6,
    }


}

# ==============================================================================
# 1. Protocol & Config
# ==============================================================================
class RadarProtocol:
    START_SIGN = b'\xff\x00\xff\x00'; STOP_SIGN = b'\xf0\x00\xf0\x00'
    HEADER_LEN = 4; FOOTER_LEN = 4; ANTENNA_INFO_LEN = 2
    FT_LEN = 32; CIR_DATA_LEN = 128; FRAME_LEN = 138

    @classmethod
    def update_protocol(cls, ft_len):
        cls.FT_LEN = ft_len
        cls.CIR_DATA_LEN = ft_len * 4
        cls.FRAME_LEN = cls.HEADER_LEN + cls.ANTENNA_INFO_LEN + cls.CIR_DATA_LEN + cls.FOOTER_LEN

    @staticmethod
    def parse_frame(frame_bytes: bytes):
        if len(frame_bytes) != RadarProtocol.FRAME_LEN: return None
        if not frame_bytes.startswith(RadarProtocol.START_SIGN): return None
        tx = frame_bytes[RadarProtocol.HEADER_LEN]; rx = frame_bytes[RadarProtocol.HEADER_LEN+1]
        cir = frame_bytes[RadarProtocol.HEADER_LEN+2 : -RadarProtocol.FOOTER_LEN]
        s16 = np.frombuffer(cir, dtype=np.int16)
        c_data = s16[0::2].astype(np.float32) + 1j * s16[1::2].astype(np.float32)
        return tx, rx, c_data

class ConfigAdapter:
    def __init__(self, dynamic_config):
        self.UDP_IP = dynamic_config.udp_ip; self.UDP_PORT = dynamic_config.udp_port
        self.UDP_TX_LIST = dynamic_config.udp_tx_list; self.UDP_RX_LIST = dynamic_config.udp_rx_list
        self.FT_LEN = dynamic_config.ft_len
        self.START_SIGN = RadarProtocol.START_SIGN; self.STOP_SIGN = RadarProtocol.STOP_SIGN

@dataclass
class RadarConfig:
    # --- 硬件与通信 ---
    ft_len: int = 32
    max_snapshots: int = 127
    udp_tx_list: List[int] = field(default_factory=lambda: [1, 2])
    udp_rx_list: List[int] = field(default_factory=lambda: [4, 5, 6, 7])
    bg_m_factor: float = 4
    num_tx_antennas: int = 2
    num_rx_antennas: int = 4
    rx_antenna_start_num: int = 4
    
    connection_mode: str = "UDP"
    serial_port: str = "COM6"
    baud_rate: int = 460800
    udp_ip: str = "127.0.0.1"
    udp_port: int = 55555
    
    # --- 天线位置 ---
    tx_positions: List[List[float]] = field(default_factory=list)
    rx_positions: List[List[float]] = field(default_factory=list)
    current_layout_name: str = "2x4_Default"

    # --- 算法参数容器 ---
    algo_params: Dict[str, Any] = field(default_factory=lambda: DEFAULT_ALGO_PARAMS.copy())
    current_algo: str = "PLOT"
    
    # --- 杂项 ---
    snapshot_rate: int = 20
    tap_interval_si: float = 1e-9 # 硬件采样间隔，通常固定
    data_save_dir: str = "./data"
    record_filename: str = "radar_capture.bin"  # <--- [新增] 默认文件名
    record_duration: float = 10.0
    playback_file_list = []
    playback_file: str = ""
    playback_duration: float = 15.0  # [新增] 期望的回放总时长（秒）
    export_pc_json: bool = False  # 新增：是否在回放时导出点云 JSON
    def load_layout(self, name):
        if name in ANTENNA_LAYOUTS:
            self.current_layout_name = name
            L = ANTENNA_LAYOUTS[name]
            self.tx_positions = L['tx']; self.rx_positions = L['rx']
            self.udp_tx_list = L['tx_list']; self.udp_rx_list = L['rx_list']
            self.num_tx_antennas = L['tx_num']; self.num_rx_antennas = L['rx_num']
            self.rx_antenna_start_num = L['rx_start']

    # --- 统一的网格定义 (X, Y) ---
    @property
    def imaging_grid(self):
        # 对应 PlotPanel extent [-1.5, 1.5, -2.5, -0.1]
        grid_x = np.linspace(-2, 2, 37)
        grid_y = np.linspace(-3, -0.1, 30)
        return grid_x, grid_y

    @property
    def TX_POS_NP(self): return np.array(self.tx_positions)
    @property
    def RX_POS_NP(self): return np.array(self.rx_positions)

# ==============================================================================
# 2. Data Manager
# ==============================================================================
class FixedBuffer:
    def __init__(self, maxlen): self.buffer = deque(maxlen=maxlen); self._maxlen = maxlen
    def append(self, item): self.buffer.append(item)
    def get_data(self): return np.array(self.buffer) if self.buffer else np.array([])
    def is_full(self): return len(self.buffer) == self._maxlen
    def __getitem__(self, index): return self.buffer[index]
    def __len__(self): return len(self.buffer)

class BackgroundRemoval:
    """实现背景去除功能的类，完全对齐用户描述逻辑。"""
    def __init__(self, M=10.0):
        """
        初始化背景去除器。
        参数:
            M (float): 指数平均的权重因子。
        """
        self.M = M
        self.cir_ref: Optional[np.ndarray] = None
        self.cir_abs_ref: Optional[np.ndarray] = None
        self.first = True

    def update(self, r: np.ndarray):
        """
        更新背景参考（复数和幅值）。
        """
        if self.first:
            self.cir_ref = r.copy()
            self.cir_abs_ref = np.abs(r)
            self.first = False
        else:
            # 指数滑动平均更新
            self.cir_ref = (1 - (1 / self.M)) * self.cir_ref + r / self.M
            self.cir_abs_ref = (1 - (1 / self.M)) * self.cir_abs_ref + np.abs(r) / self.M

    def remove_background(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从信号中去除背景。
        返回: (去背景后的复数信号, 去背景后的幅值信号)
        """
        if self.first:
            self.update(r)
            return r, np.abs(r)
        else:
            # 计算当前帧与之前参考的差值
            r_no_bg = r - self.cir_ref
            r_abs_no_bg = np.abs(r) - self.cir_abs_ref
            
            # 计算完差值后，更新参考供下一帧使用
            self.update(r)
            return r_no_bg, r_abs_no_bg

class RadarDataManager:
    def __init__(self, config: RadarConfig):
        self.config = config
        self.pairs = [(tx, rx) for tx in config.udp_tx_list for rx in config.udp_rx_list]
        self.snapshots_data = {p: {'raw': FixedBuffer(config.max_snapshots), 'complex': FixedBuffer(config.max_snapshots), 'abs': FixedBuffer(config.max_snapshots)} for p in self.pairs}
        self.bgs = {p: BackgroundRemoval(M=config.bg_m_factor) for p in self.pairs}
        self.buffer_full = False

    def process_frame(self, tx, rx, raw):
        if (tx, rx) not in self.snapshots_data: return
        self.snapshots_data[(tx, rx)]['raw'].append(raw)
        self.bgs[(tx, rx)].M = self.config.bg_m_factor
        

        # -------------------------------------------------------------

        # 2. 调用对齐后的背景去除逻辑
        # 注意：这里返回的 r_abs_no_bg 是 |r| - ref_abs
        r_no_bg, r_abs_no_bg = self.bgs[(tx, rx)].remove_background(raw)

        # 3. 去背景后的复数送入 'complex'
        self.snapshots_data[(tx, rx)]['complex'].append(r_no_bg)
        
        # 4. 去背景后的幅值送入 'abs'
        self.snapshots_data[(tx, rx)]['abs'].append(r_abs_no_bg)
        
        if self.snapshots_data[self.pairs[0]]['complex'].is_full(): 
            self.buffer_full = True

    def get_all_snapshot_as_array(self, type='complex'):
        if not self.buffer_full: return None
        
        # 1. 确定当前所有缓存中的最小有效长度
        # 这是为了防止某些通道只有 126 帧，而另一些有 127 帧导致的广播错误
        all_buffs = [b[type] for b in self.snapshots_data.values()]
        actual_lens = [len(b) for b in all_buffs]
        min_len = min(actual_lens) if actual_lens else 0
        
        # 2. 如果是回放模式且数据量不足，我们强制对齐到最小长度
        # 这样 arr 的最后一个维度将动态匹配实际抓取到的快照数
        target_len = self.config.max_snapshots
        if self.config.connection_mode == 'PLAYBACK' and min_len < target_len:
            target_len = min_len

        if target_len <= 0: return None

        dt = np.complex64 if type=='complex' else np.float32
        # 使用确定的 target_len 初始化数组
        arr = np.zeros((self.config.num_rx_antennas, self.config.num_tx_antennas, self.config.ft_len, target_len), dtype=dt)
        
        tx_map = {v: i for i, v in enumerate(self.config.udp_tx_list)}
        rx_map = {v: i for i, v in enumerate(self.config.udp_rx_list)}
        
        for (tx, rx), buffs in self.snapshots_data.items():
            if tx in tx_map and rx in rx_map:
                # 3. 提取数据时进行切片 [:target_len]，确保形状完全匹配
                data_slice = buffs[type].get_data()[:target_len].T
                arr[rx_map[rx], tx_map[tx], :, :] = data_slice
                
        return arr

# ==============================================================================
# 3. Algorithm Processor (Logic Ported + Vectorized)
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
        
        # 1. 计算原始角度 (Radiants)
        if method == "FFT":
            angle = -self.calculate_fft_aoa(phase_vec, params)
        elif method == "MUSIC":
            angle = -self.calculate_music_aoa(phase_vec, params)
        else:
            angle = 0.0

        # 2. 仅在 PLAYBACK 模式下应用线性校准
        if self.config.connection_mode == 'PLAYBACK':
            # 获取校准参数
            offset_deg = params.get("aoa_offset_deg", 0.0)
            scale = params.get("aoa_scale", 1.0)
            
            # 将角度偏移从“度”转换为“弧度”
            offset_rad = np.deg2rad(offset_deg)
            
            # 执行线性变换: y = kx + b
            angle = (angle * scale) + offset_rad
            
            # 可选：限制角度范围在 [-pi/2, pi/2]
            angle = np.clip(angle, -np.pi/2, np.pi/2)
            
        return angle
    
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

            # AoA 角度估算
            az_angle = self.estimate_aoa(phase_vec_selected, params)
            # 5. 坐标转换与点云构建
            # Range = 索引 * 分辨率
            dist = (r_idx+cir_offset - 6) * params['dist_per_tap']

            # 笛卡尔坐标映射: x(横向), y(纵向)
            point_x = dist *  np.sin(az_angle)
            point_y = -dist * np.cos(az_angle) # 纵向深度
            # point_z = dist * np.sin(el_angle) # 目标相对于雷达平面的高度
            # 计算 SNR 用于可视化强度
            snr_val = 10 * np.log10(power_map[r_idx, d_idx] / (noise_avg[r_idx, d_idx] + 1e-12))
            detected_points.append({
            'pos': (point_x, point_y),
            'snr': snr_val,
            'time': time.time() # 记录时间戳用于“出现后消失”效果
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

    def map_to_2d_grid(self, raw_vec):
        """
        将原始 8 通道数据映射到 2x5 虚拟面阵网格
        raw_vec 顺序: [T0T0, T0T1, T0R0, T0R1, T1T0, T1T1, T1R0, T1R1]
        """
        # 初始化 2x5 的复数矩阵
        grid = np.zeros((2, 5), dtype=complex)
        
        # 冗余点处理：T0-T1 (idx 1) 和 T1-T0 (idx 4) 物理位置重合
        redundant_avg = (raw_vec[1] + raw_vec[4]) / 2.0
        
        # 填充 Row 0 (Y=0, 上排)
        grid[0, 0] = raw_vec[0]          # T0-T0 (0,0)
        grid[0, 2] = redundant_avg       # (1.0, 0)
        grid[0, 4] = raw_vec[5]          # T1-T1 (2.0, 0)
        
        # 填充 Row 1 (Y=-0.5, 下排)
        grid[1, 0] = raw_vec[2]          # T0-R0 (0, -0.5)
        grid[1, 1] = raw_vec[3]          # T0-R1 (0.5, -0.5)
        grid[1, 2] = raw_vec[6]          # T1-R0 (1.0, -0.5)
        grid[1, 3] = raw_vec[7]          # T1-R1 (1.5, -0.5)
        
        return grid






# ==============================================================================
# 4. IO Layer
# ==============================================================================
class LiveRadarSource:
    def __init__(self, config: RadarConfig):
        self.config = config; self.running = False; self.thread = None
        self.data_queue = queue.Queue(maxsize=5000); self.udp_server = None
        self.is_recording = False; self.record_file = None; self.record_lock = threading.Lock()
        self.rec_start_time = 0; self.rec_duration_target = 0; self.measured_fps = 0.0; self.last_fps_time = time.time(); self.frame_count_sec = 0
    def start(self):
        self.running = True; RadarProtocol.update_protocol(self.config.ft_len)
        if self.config.connection_mode == 'UDP': self.udp_server = UDPFrameServer(ConfigAdapter(self.config)); self.udp_server.start()
        self.thread = threading.Thread(target=self._io_loop, daemon=True); self.thread.start(); return True
    def stop(self):
        self.running = False; self.stop_recording()
        if self.thread: self.thread.join()
        if self.udp_server: self.udp_server.stop()
    def start_recording(self, filename, duration=0):
        with self.record_lock:
            try: os.makedirs(os.path.dirname(filename), exist_ok=True); self.record_file = open(filename, 'wb'); self.is_recording = True; self.rec_start_time = time.time(); self.rec_duration_target = duration; meta = asdict(self.config); meta['recorded_date'] = str(datetime.now()); json.dump(meta, open(filename + ".meta", 'w'), indent=4); return True
            except: return False
    def stop_recording(self):
        with self.record_lock:
            if self.is_recording and self.record_file:
                try: m = json.load(open(self.record_file.name+".meta")); m['actual_fps'] = self.measured_fps; json.dump(m, open(self.record_file.name+".meta",'w'), indent=4)
                except: pass
                self.is_recording = False; self.record_file.close(); self.record_file = None
    def _io_loop(self):
        ser = None
        if self.config.connection_mode == 'SERIAL': 
            try: ser = serial.Serial(self.config.serial_port, self.config.baud_rate, timeout=0.05)
            except: self.running = False; return
        while self.running:
            now = time.time()
            if self.is_recording and self.rec_duration_target > 0 and now - self.rec_start_time >= self.rec_duration_target: self.stop_recording()
            if now - self.last_fps_time >= 1.0: self.measured_fps = self.frame_count_sec / (now - self.last_fps_time); self.frame_count_sec = 0; self.last_fps_time = now
            raw_chunk = b''
            try:
                if self.config.connection_mode == 'UDP': f = self.udp_server.get_frame(); raw_chunk = f if f else b''; time.sleep(0.002) if not f else None
                elif self.config.connection_mode == 'SERIAL': raw_chunk = ser.read(ser.in_waiting) if ser.in_waiting else b''; time.sleep(0.002) if not raw_chunk else None
            except: pass
            if not raw_chunk: continue
            if self.is_recording and self.record_file: self.record_file.write(raw_chunk)
            parsed = RadarProtocol.parse_frame(raw_chunk)
            if parsed: self.frame_count_sec += 1; self.data_queue.put(parsed) if not self.data_queue.full() else None
        if ser: ser.close()
    def get_batch_frames(self):
        frames = []; 
        try: 
            while True: frames.append(self.data_queue.get_nowait())
        except queue.Empty: pass
        return frames
    def get_rec_status(self): return self.is_recording, (time.time() - self.rec_start_time if self.is_recording else 0)
    def get_progress(self): return 0

class FilePlaybackSource:
    def __init__(self, config: RadarConfig):
        self.config = config; 
        self.running = False; 
        self.thread = None; 
        self.data_queue = queue.Queue(maxsize=5000); 
        self.total_bytes = 0; 
        self.read_bytes = 0; 
        self.progress = 0.0; 
        self.playback_fps = config.snapshot_rate
    def start(self):
        if not self.config.playback_file or not os.path.exists(self.config.playback_file):
            return False
        
        # [修改] 不再依赖 meta 文件更新配置，直接使用当前 config 中的 ft_len
        RadarProtocol.update_protocol(self.config.ft_len)
        
        self.total_bytes = os.path.getsize(self.config.playback_file)
        self.read_bytes = 0
        
        # [新增] 根据目标时长计算 FPS
        # 总帧数 = 总字节数 / 单帧长度
        total_frames = self.total_bytes / RadarProtocol.FRAME_LEN
        if self.config.playback_duration > 0:
            self.playback_fps = total_frames / self.config.playback_duration
        else:
            self.playback_fps = self.config.snapshot_rate # 默认 fallback
            
        print(f"[Playback] Total Frames: {total_frames:.0f}, Target Duration: {self.config.playback_duration}s, Calculated FPS: {self.playback_fps:.2f}")

        self.running = True
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _play_loop(self):
        flen = RadarProtocol.FRAME_LEN
        # 根据计算出的 FPS 确定每帧间隔
        interval = 1.0 / self.playback_fps if self.playback_fps > 0 else 0.05
        
        with open(self.config.playback_file, 'rb') as f:
            while self.running:
                t0 = time.time()
                chunk = f.read(flen)
                if len(chunk) < flen:
                    self.running = False
                    break
                
                self.read_bytes += len(chunk)
                self.progress = (self.read_bytes / self.total_bytes) * 100
                
                parsed = RadarProtocol.parse_frame(chunk)
                if parsed:
                    if not self.data_queue.full():
                        self.data_queue.put(parsed)
                
                # 精确控制播放速度
                dt = time.time() - t0
                if interval > dt:
                    time.sleep(interval - dt)
    def get_batch_frames(self):
        frames = []; 
        try: 
            while True: frames.append(self.data_queue.get_nowait())
        except queue.Empty: pass
        return frames
    def get_rec_status(self): return False, 0
    def get_fps(self): return self.playback_fps
    def get_progress(self): return self.progress

# ==============================================================================
# 5. UI
# ==============================================================================
class AlgoSettingsDialog(tk.Toplevel):
    def __init__(self, parent, algo_name, params_dict, callback):
        super().__init__(parent)
        self.title(f"Settings: {algo_name}")
        # --- 修改点 1: 取消 geometry，改用 minsize，让窗口根据内容自适应 ---
        self.minsize(350, 450) 
        self.params = params_dict.copy()
        self.callback = callback
        self.vars = {}
        self._build_ui()

    def _build_ui(self):
        canvas = tk.Canvas(self); scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview); frm = tk.Frame(canvas)
        frm.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frm, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for r, (key, val) in enumerate(self.params.items()):
            tk.Label(frm, text=key+":", anchor='w').grid(row=r, column=0, sticky='w', pady=2, padx=5)
            if isinstance(val, bool): 
                var = tk.BooleanVar(value=val); widget = tk.Checkbutton(frm, variable=var)
            else: 
                var = tk.StringVar(value=str(val)); widget = tk.Entry(frm, textvariable=var)
            self.vars[key] = var
            widget.grid(row=r, column=1, sticky='ew', padx=5)
        
        # 使 Entry 列可以随窗口拉伸
        frm.columnconfigure(1, weight=1)

        # --- 修改点 2: 调整按钮布局 ---
        btn_frame = tk.Frame(self, pady=15, bg='#f0f0f0') # 加一点背景色区分
        btn_frame.pack(side="bottom", fill='x')
        
        # 改为靠左排列 (side='left')，这样无论窗口多窄，按钮始终可见
        # 或者增加 padx 让它们在中间偏左一点
        tk.Button(btn_frame, text="  Save  ", command=self._save, bg='#cfc', width=10).pack(side='left', padx=(20, 10))
        tk.Button(btn_frame, text=" Cancel ", command=self.destroy, width=10).pack(side='left', padx=10)
    def _save(self):
        new_params = {}
        for key, var in self.vars.items():
            orig_val = self.params[key]
            try:
                val = var.get()
                if isinstance(orig_val, bool): 
                    new_params[key] = bool(val)
                elif isinstance(orig_val, list): 
                    # 更好地处理列表输入，支持 [0,1] 或 0,1 格式
                    s = val.strip()
                    if not s.startswith('['): s = '[' + s + ']'
                    new_params[key] = json.loads(s)
                elif isinstance(orig_val, int): 
                    new_params[key] = int(val)
                elif isinstance(orig_val, float): 
                    new_params[key] = float(val)
                else: 
                    new_params[key] = val
            except Exception as e:
                print(f"参数 {key} 转换失败: {e}")
                new_params[key] = orig_val # 转换失败则保持原值
        
        # 执行回调更新 config 并触发 App.update_layout
        self.callback(new_params); 
        self.destroy()

class PlotPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent); 
        self.figure = plt.Figure(figsize=(10, 6), dpi=100); 
        self.canvas = FigureCanvasTkAgg(self.figure, self); 
        self.canvas.get_tk_widget().pack(fill='both', expand=True); 
        self.axes = {}; 
        self.plots = {}
        self.pc_history = []  # 用于存储当前活跃的点云缓存    
    def init_layout(self, mode, params):
        self.figure.clf(); self.axes = {}; self.plots = {}
        if mode == 'PLOT':
            ax = self.figure.add_subplot(111); ax.set_title("Waveform"); ax.grid(True)
            self.plots['line_abs'], = ax.plot([], [], 'r-', label='Abs'); self.plots['line_real'], = ax.plot([], [], 'b-', alpha=0.5, label='Real'); self.plots['line_imag'], = ax.plot([], [], 'g-', alpha=0.5, label='Imag')
            ax.axhline(32767, c='k', ls='--'); ax.axhline(-32768, c='k', ls='--')
            self.plots['text_sat'] = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='red', fontweight='bold'); self.plots['text_max'] = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='blue', fontweight='bold')
            ax.legend(); ax.set_ylim(params.get('ylim_min',-100), params.get('ylim_max',100)); self.axes['main'] = ax
        elif mode == '2D-MUSIC':
            gs = self.figure.add_gridspec(1, 3)
            ax1 = self.figure.add_subplot(gs[0,0])
            ax2 = self.figure.add_subplot(gs[0,1])
            is_polar = (params.get('fusion_mode') == 'polar')
            ax3 = self.figure.add_subplot(gs[0,2], projection='polar' if is_polar else None)
            
            # 物理坐标映射: X: -1.5~1.5m, Y: -2.5~-0.1m
            extent = [-2, 2, -3, -0.1]
            ax1.set_title("2D-MUSIC (Meters)")
            ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
            
            self.plots['music_im'] = ax1.imshow(
                np.zeros((30,37)), 
                aspect='auto', 
                origin='lower', 
                cmap='jet',
                extent=extent,
                interpolation='bilinear'
            )
            
            ax2.set_title("Range-Doppler")
            self.plots['dop_im'] = ax2.imshow(np.zeros((32,64)), aspect='auto', origin='lower', cmap='jet'); self.plots['dop_line'], = ax2.plot([],[],'r--',lw=1)
            
            ax3.set_title("Fusion")
            self.plots['tgt'], = ax3.plot([],[],'ro',ms=12)
            if not is_polar: 
                ax3.grid(True); ax3.set_xlim(-params.get('plot_xlim', 2), params.get('plot_xlim', 2)); ax3.set_ylim(params.get('plot_ylim_min', -4), params.get('plot_ylim_max', 0))
                ax3.add_patch(plt.Rectangle((-0.65, -1.8), 1.3, 1.8, ec='blue', fc='none', lw=2))
            else: ax3.set_rmax(3.5); ax3.set_theta_zero_location('S')
            self.axes = {'music': ax1, 'dop': ax2, 'fus': ax3}
        elif mode == 'POINT-CLOUD':
            ax = self.figure.add_subplot(111)
            # 按照论文表 II 设置坐标范围：X 为 +/- 1.5m，Y 为 -2.5m 到 0m [cite: 332, 345]
            ax.set_xlim(-params.get('plot_xlim', 1.5), params.get('plot_xlim', 1.5))
            ax.set_ylim(params.get('plot_ylim_min', -2.5), params.get('plot_ylim_max', 0))
            ax.set_title("Vehicle Occupancy Point Cloud (CA-CFAR)")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # 初始化散点图对象
            self.plots['pc_scatter'] = ax.scatter([], [], c=[], cmap='cool', s=30, alpha=0.8)
            
            # 绘制座椅椭圆参考线（根据论文表 II 几何参数 [cite: 332]）
            self._draw_seating_ellipses(ax, params)
            self.axes['main'] = ax
        self.canvas.draw()


    def update_data(self, mode, data):
        if not data: return
        p = data['params']
        if mode == 'PLOT':
            y = data['y']; x = np.arange(len(y)); self.plots['line_abs'].set_data(x, y)
            if p.get('show_raw'): self.plots['line_real'].set_data(x, data['y_real']); self.plots['line_imag'].set_data(x, data['y_imag']); self.plots['line_real'].set_visible(True); self.plots['line_imag'].set_visible(True)
            else: self.plots['line_real'].set_visible(False); self.plots['line_imag'].set_visible(False)
            self.axes['main'].set_title(f"Waveform: TX{data['pair'][0]}-RX{data['pair'][1]}"); self.plots['text_max'].set_text(f"Max: {data['max_real']:.0f}"); self.plots['text_sat'].set_text("SAT!" if data['is_sat'] else "")
            self.axes['main'].set_ylim(p['ylim_min'], p['ylim_max']); self.axes['main'].set_xlim(0, len(y))
        elif mode == '2D-MUSIC':
            # MUSIC 图像更新与自动对比度
            m_img = data['music_img']
            self.plots['music_im'].set_data(m_img)

            # --- 2. 计算最大值及其坐标 ---
            max_val = np.max(m_img)
            if max_val > 0:
                # 找到最大值的行列索引 (row, col)
                max_idx = np.unravel_index(np.argmax(m_img), m_img.shape)
                row, col = max_idx
                
                # 映射到物理坐标 (根据 imshow 的 extent [-1.5, 1.5, -2.5, -0.1])
                # X: col 0 -> -1.5, col 36 -> 1.5
                # Y: row 0 -> -2.5, row 29 -> -0.1 (origin='lower')
                rows, cols = m_img.shape
                max_x = -1.5 + (col / (cols - 1)) * 3.0
                max_y = -2.5 + (row / (rows - 1)) * 2.4
                
                # 更新标题
                self.axes['music'].set_title(f"2D-MUSIC | Max: {max_val:.2f} @ ({max_x:.2f}m, {max_y:.2f}m)")
            else:
                self.axes['music'].set_title("2D-MUSIC | No Signal")

            vmax_cfg = p.get('music_vmax', 0)
            if vmax_cfg > 0:
                # 使用用户手动设置的固定上限
                self.plots['music_im'].set_clim(vmin=0, vmax=vmax_cfg)
            elif np.max(m_img) > 0:
                # 如果设置为 0，则退回到原有的自动对比度模式
                self.plots['music_im'].set_clim(vmin=0, vmax=np.max(m_img))

            # Doppler 图像更新
            d_img = data['doppler_img']
            self.plots['dop_im'].set_data(d_img)
            if np.max(d_img) > 0:
                self.plots['dop_im'].set_clim(vmin=0, vmax=np.max(d_img))

            if data['peak_tap'] != -1: self.plots['dop_line'].set_data([0, 63], [data['peak_tap']]*2)
            else: self.plots['dop_line'].set_data([], [])
            
            if data['has_target']:
                tx, ty = data['target_pos']
                if p['fusion_mode'] == 'polar': self.plots['tgt'].set_data([np.arctan2(tx, -ty)], [np.sqrt(tx**2+ty**2)])
                else: self.plots['tgt'].set_data([tx], [ty])
            else: self.plots['tgt'].set_data([], [])
            self.axes['fus'].set_title(f"Breath: {data['breath_val']:.2f} | Motion: {data['max_motion']:.2f}")
        elif mode == 'POINT-CLOUD':
            if not data or 'detected_points' not in data: return
            p = data['params']
            breath_val = data.get('breath_val', 0.0)
            # --- 新增：动态更新标题逻辑 ---
            base_title = "Vehicle Occupancy Point Cloud (CA-CFAR)"
            info_text = f"Breath: {breath_val:.3f}"
            if 'filename' in data:
                # 在回放模式下显示文件名
                self.axes['main'].set_title(f"{base_title} ({info_text})\nFile: {data['filename']}", fontsize=10)
            else:
                # 实时模式下显示默认标题
                self.axes['main'].set_title(f"{base_title}\n{info_text}")
            # ----------------------------

            current_time = time.time()
            lifetime = p.get('point_lifetime_sec', 1.5)
            # 1. 将新探测到的点加入历史缓存
            for pt in data['detected_points']:
                self.pc_history.append(pt)
            # 2. 过滤过期点并计算透明度
            still_active = []
            x_coords, y_coords, alphas, colors = [], [], [], []
            for pt in self.pc_history:
                age = current_time - pt['time']
                if age < lifetime:
                    # 计算随时间衰减的 alpha 值
                    alpha = max(0.1, 1.0 - (age / lifetime))
                    still_active.append(pt)
                    x_coords.append(pt['pos'][0])
                    y_coords.append(pt['pos'][1])
                    alphas.append(alpha)
                    colors.append(pt['snr']) # 颜色深浅代表信噪比 [cite: 280, 285]

            self.pc_history = still_active
            # 3. 批量更新散点图（这种方式比逐点绘制更流畅）
            if x_coords:
                # 更新偏移量和颜色
                self.plots['pc_scatter'].set_offsets(np.c_[x_coords, y_coords])
                self.plots['pc_scatter'].set_array(np.array(colors))
                # 批量更新透明度
                self.plots['pc_scatter'].set_alpha(alphas)
            else:
                self.plots['pc_scatter'].set_offsets(np.empty((0, 2)))

            self.canvas.draw_idle() # 使用 draw_idle 提高响应速度
        self.canvas.draw()

    def _draw_seating_ellipses(self, ax, params):
        """
        根据论文 Table II 的几何参数绘制座椅参考椭圆 [cite: 332, 345]
        """
        # 获取车辆配置（4座或5座）
        is_5_seater = (params.get("seat_type") == "5_seats")
        
        # 论文 Table II 核心参数:
        # 前排中心 y = -0.6m, 后排中心 y = -0.6 - 0.8 = -1.4m
        # 左右间距 0.6m -> x = ±0.3m
        # 半长轴 (Semi-axis) 大约为 0.2m
        
        if not is_5_seater:
            # 4座车布局 [cite: 169, 332]
            seats = [
                {'center': (-0.3, -0.6), 'name': '1'}, # Driver
                {'center': (0.3, -0.6),  'name': '2'}, # Passenger
                {'center': (-0.3, -1.4), 'name': '3'}, # Rear-L
                {'center': (0.3, -1.4),  'name': '4'}  # Rear-R
            ]
            rx, ry = 0.2, 0.2 # 椭圆半径 
        else:
            # 5座车布局 [cite: 169, 332]
            seats = [
                {'center': (-0.3, -0.5), 'name': '1'},
                {'center': (0.3, -0.5),  'name': '2'},
                {'center': (-0.4, -1.3), 'name': '3'},
                {'center': (0, -1.3),    'name': '5'}, # 后排中间
                {'center': (0.4, -1.3),  'name': '4'}
            ]
            rx, ry = 0.25, 0.2 # 5座车前排稍宽 

        for s in seats:
            # 使用论文公式 (12) 定义的区域绘制椭圆 
            ellipse = patches.Ellipse(
                s['center'], width=rx*2, height=ry*2,
                edgecolor='red', facecolor='none', 
                linestyle='--', linewidth=1.5, alpha=0.7
            )
            ax.add_patch(ellipse)
            ax.text(s['center'][0], s['center'][1], s['name'], color='red', ha='center')        

class ControlPanel(tk.Frame):
    def __init__(self, parent, config, cbs):
        super().__init__(parent, width=320, bg='#f0f0f0')
        self.config = config
        self.cbs = cbs
        self.pack_propagate(False)

        # --- 标题 ---
        tk.Label(self, text="Radar V22 (Fixed)", bg='#f0f0f0', font=('Arial', 12, 'bold')).pack(pady=5)

        # --- 模式选择 ---
        frm_mode = tk.Frame(self, bg='#f0f0f0')
        frm_mode.pack(fill='x', padx=5, pady=5)
        self.mode_var = tk.StringVar(value=config.connection_mode)
        ttk.Combobox(frm_mode, textvariable=self.mode_var, values=('UDP', 'SERIAL', 'PLAYBACK'), state='readonly').pack(fill='x')
        self.mode_var.trace_add('write', self._on_mode_change)

        # --- 基础配置区域 (折叠或简化显示) ---
        # 只保留最常用的配置，避免界面太长
        for title, attrs in [
            ("Data Structure", ["ft_len", "udp_tx_list", "udp_rx_list"]),
            ("Communication", ["serial_port", "baud_rate", "udp_port"])
        ]:
            frm = tk.LabelFrame(self, text=title, bg='#f0f0f0')
            frm.pack(fill='x', padx=5, pady=2)
            for attr in attrs:
                self._entry(frm, attr.replace('_', ' ').title() + ":", attr)

        # --- 算法设置 ---
        frm_algo = tk.LabelFrame(self, text="Algo & Geometry", bg='#f0f0f0', fg='purple')
        frm_algo.pack(fill='x', padx=5, pady=5)
        
        row1 = tk.Frame(frm_algo, bg='#f0f0f0')
        row1.pack(fill='x', padx=2)
        tk.Label(row1, text="Algo:", bg='#f0f0f0').pack(side='left')
        self.cb_algo = ttk.Combobox(row1, values=('PLOT', '2D-MUSIC','POINT-CLOUD'), width=10, state='readonly')
        self.cb_algo.set(config.current_algo)
        self.cb_algo.pack(side='left')
        self.cb_algo.bind("<<ComboboxSelected>>", self._on_algo_change)
        tk.Button(row1, text="⚙️ Params", command=self._open_algo).pack(side='right')
        
        row2 = tk.Frame(frm_algo, bg='#f0f0f0')
        row2.pack(fill='x', padx=2)
        tk.Label(row2, text="Layout:", bg='#f0f0f0').pack(side='left')
        self.cb_layout = ttk.Combobox(row2, values=list(ANTENNA_LAYOUTS.keys()), state='readonly')
        self.cb_layout.set(config.current_layout_name)
        self.cb_layout.pack(side='right', fill='x', expand=True)
        self.cb_layout.bind("<<ComboboxSelected>>", self._on_layout_change)

        # --- [新增] 录制设置面板 ---
        frm_rec = tk.LabelFrame(self, text="Recording Settings", bg='#f0f0f0', fg='blue')
        frm_rec.pack(fill='x', padx=5, pady=5)

        # 1. 保存路径
        row_path = tk.Frame(frm_rec, bg='#f0f0f0')
        row_path.pack(fill='x', padx=2, pady=2)
        tk.Label(row_path, text="Save To:", bg='#f0f0f0', width=8, anchor='w').pack(side='left')
        self.var_save_dir = tk.StringVar(value=config.data_save_dir)
        tk.Entry(row_path, textvariable=self.var_save_dir, font=('Arial', 8)).pack(side='left', fill='x', expand=True)
        tk.Button(row_path, text="...", width=3, command=self._choose_dir).pack(side='right')

        # 2. [新增] 文件名 (File Name)
        row_fname = tk.Frame(frm_rec, bg='#f0f0f0')
        row_fname.pack(fill='x', padx=2, pady=2)
        tk.Label(row_fname, text="File Name:", bg='#f0f0f0', width=9, anchor='w').pack(side='left')
        # 如果配置里没有文件名，给一个带时间戳的默认值
        default_name = config.record_filename if config.record_filename else f"radar_{datetime.now():%H%M%S}.bin"
        self.var_save_name = tk.StringVar(value=default_name)
        tk.Entry(row_fname, textvariable=self.var_save_name, font=('Arial', 8)).pack(side='left', fill='x', expand=True)


        # 3. 时长设置
        row_dur = tk.Frame(frm_rec, bg='#f0f0f0')
        row_dur.pack(fill='x', padx=2, pady=2)
        tk.Label(row_dur, text="Dur(s):", bg='#f0f0f0', width=8, anchor='w').pack(side='left')
        self.var_rec_dur = tk.DoubleVar(value=config.record_duration)
        tk.Entry(row_dur, textvariable=self.var_rec_dur).pack(side='left', fill='x', expand=True)
        tk.Label(row_dur, text="(0=Inf)", bg='#f0f0f0', fg='gray').pack(side='right')

        # 4. 录制按钮
        self.btn_rec = tk.Button(frm_rec, text="Start Recording", command=self._rec, bg='#ddd')
        self.btn_rec.pack(fill='x', padx=5, pady=5)

        # --- 回放控制 ---
        self.frm_pb = tk.Frame(self, bg='#f0f0f0')
        self.frm_pb.pack(fill='x', padx=5)
        tk.Button(self.frm_pb, text="Select Playback File...", command=self._sel_pb).pack(fill='x')
        self.lbl_pb = tk.Label(self.frm_pb, text="None", bg='#ddd', anchor='w')
        self.lbl_pb.pack(fill='x')
        
        self.var_downsample = tk.BooleanVar(value=False)
        chk_downsample = tk.Checkbutton(
            self.frm_pb, 
            text="Playback Downsample (1/2 FPS)", 
            variable=self.var_downsample,
            bg='#f0f0f0',
            command=self._sync_downsample_config
        )
        chk_downsample.pack(fill='x', padx=5)

        # 2. [新增] 播放时长设置
        row_pb_dur = tk.Frame(self.frm_pb, bg='#f0f0f0')
        row_pb_dur.pack(fill='x', padx=2, pady=2)
        tk.Label(row_pb_dur, text="Play Dur(s):", bg='#f0f0f0', width=10, anchor='w').pack(side='left')
        self.var_pb_dur = tk.DoubleVar(value=config.playback_duration)
        tk.Entry(row_pb_dur, textvariable=self.var_pb_dur).pack(side='left', fill='x', expand=True)
        # [新增] 导出 JSON 开关
        self.var_export_json = tk.BooleanVar(value=config.export_pc_json)
        chk_export = tk.Checkbutton(
            self.frm_pb, 
            text="Export PC to JSON (Playback)", 
            variable=self.var_export_json,
            bg='#f0f0f0',
            command=self._sync_export_config
        )
        chk_export.pack(fill='x', padx=5)

        # --- [新增] 进度条组件 ---
        self.pb_bar = ttk.Progressbar(self.frm_pb, orient='horizontal', mode='determinate')
        self.pb_bar.pack(fill='x', padx=5, pady=5)

        # --- [新增] 进度百分比文字 ---
        self.lbl_prog_text = tk.Label(self.frm_pb, text="Progress: 0.0%", bg='#f0f0f0', font=('Arial', 8))
        self.lbl_prog_text.pack(fill='x')

        # 绑定同步到 config
        self.var_pb_dur.trace_add("write", lambda *args: self._sync_pb_dur())

        # --- 系统控制 ---
        tk.Frame(self, height=10, bg='#f0f0f0').pack() # Spacer
        self.lbl_status = tk.Button(self, text="Ready", state='disabled', bg='#ccc')
        self.lbl_status.pack(fill='x', padx=5)
        
        self.btn_start = tk.Button(self, text="START SYSTEM", command=cbs['start'], height=2, bg='#cfc')
        self.btn_start.pack(fill='x', padx=5, pady=2)
        
        self.btn_stop = tk.Button(self, text="STOP SYSTEM", command=cbs['stop'], state='disabled', bg='#fcc')
        self.btn_stop.pack(fill='x', padx=5, pady=2)

        self.recording_state = False
        self._on_mode_change()
    def _sync_downsample_config(self):
        self.config.downsample_pb = self.var_downsample.get()
    # [新增] 同步函数

    def _sync_export_config(self):
        self.config.export_pc_json = self.var_export_json.get()
    def _sync_pb_dur(self):
        try:
            self.config.playback_duration = float(self.var_pb_dur.get())
        except:
            pass

    def _entry(self, p, l, a):
        r = tk.Frame(p, bg='#f0f0f0')
        r.pack(fill='x')
        tk.Label(r, text=l, width=14, anchor='w', bg='#f0f0f0').pack(side='left')
        val = getattr(self.config, a)
        v = tk.StringVar(value=str(val) if not isinstance(val, list) else ",".join(map(str, val)))
        e = tk.Entry(r, textvariable=v)
        e.pack(side='right', expand=True, fill='x')
        def _save(*_):
            try:
                raw = v.get()
                if isinstance(val, list):
                    setattr(self.config, a, [int(x) for x in raw.replace('，', ',').split(',') if x.strip()])
                elif isinstance(val, int):
                    setattr(self.config, a, int(raw))
                elif isinstance(val, float):
                    setattr(self.config, a, float(raw))
                else:
                    setattr(self.config, a, raw)
            except: pass
        v.trace_add("write", _save)

    def _choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.config.data_save_dir)
        if d:
            self.config.data_save_dir = d
            self.var_save_dir.set(d)

    def _on_mode_change(self, *_):
        self.config.connection_mode = self.mode_var.get()
        is_pb = (self.config.connection_mode == 'PLAYBACK')
        
        if is_pb:
            self.btn_rec.config(state='disabled')
            self.frm_pb.pack(fill='x', padx=5, pady=5)
            # 切换到回放模式时重置进度条
            self.pb_bar['value'] = 0
            self.lbl_prog_text.config(text="Progress: 0.0%")
        else:
            self.btn_rec.config(state='normal')
            self.frm_pb.pack_forget()

    def _on_layout_change(self, e): 
        self.config.load_layout(self.cb_layout.get())

    def _on_algo_change(self, e): 
        self.config.current_algo = self.cb_algo.get()
        self.cbs['update_layout'](self.config.current_algo)

    def _open_algo(self): 
        AlgoSettingsDialog(self, self.config.current_algo, 
                           self.config.algo_params.get(self.config.current_algo, {}), 
                           lambda p: (self.config.algo_params.update({self.config.current_algo: p}), 
                                      self.cbs['update_layout'](self.config.current_algo)))

    def _sel_pb(self): 
        # 修改为 askopenfilenames (复数)
        files = filedialog.askopenfilenames(filetypes=[("Radar Bin", "*.bin"), ("All", "*.*")])
        if files:
            # 将选中的文件列表存入 config (需确保 config 有这个属性)
            self.config.playback_file_list = list(files) 
            self.config.playback_file = files[0] # 默认第一个
            count = len(files)
            self.lbl_pb.config(text=f"Selected {count} files") # UI 显示数量

    def _rec(self):
        if not self.recording_state:
            # --- 开始录制 ---
            try:
                # 1. 获取并校验路径
                save_dir = self.var_save_dir.get().strip()
                self.config.data_save_dir = save_dir
                
                # 2. 获取并校验文件名
                fname = self.var_save_name.get().strip()
                if not fname:
                    fname = f"radar_{datetime.now():%Y%m%d_%H%M%S}.bin"
                
                # 自动补全 .bin 后缀
                if not fname.lower().endswith('.bin'):
                    fname += ".bin"
                
                self.config.record_filename = fname
                self.var_save_name.set(fname) # 更新UI显示

                self.config.record_duration = float(self.var_rec_dur.get())
            except ValueError:
                messagebox.showerror("Error", "无效的参数输入")
                return

            # 3. 检查并创建目录
            if not os.path.exists(self.config.data_save_dir):
                try: os.makedirs(self.config.data_save_dir)
                except: messagebox.showerror("Error", "无法创建保存目录"); return

            # 4. 拼接完整路径
            full_path = os.path.join(self.config.data_save_dir, self.config.record_filename)
            
            # 5. 调用后端开始录制
            if self.cbs['rec_start'](full_path, self.config.record_duration):
                self.recording_state = True
                self.btn_rec.config(bg='#f88', text="Stop Recording")
        else:
            # --- 停止录制 ---
            self.cbs['rec_stop']()
            self.recording_state = False
            self.btn_rec.config(bg='#ddd', text="Start Recording")

    def update_ui(self, run, rec, rec_time, fps, pb_prog):
        # 系统状态
        self.lbl_status.config(text=f"Running ({fps:.1f} FPS)" if run else "Stopped", bg='#8f8' if run else '#ccc')
        self.btn_start.config(state='disabled' if run else 'normal')
        self.btn_stop.config(state='normal' if run else 'disabled')
        
        # 录制按钮状态逻辑
        if self.config.connection_mode != 'PLAYBACK':
            # 如果系统未运行，通常不允许录制，或者允许录制空数据？通常是不允许
            self.btn_rec.config(state='normal' if run else 'disabled')
            
            # 检测实际录制状态 (rec 是从 backend 传来的真实状态)
            if rec:
                self.recording_state = True
                target = self.config.record_duration
                if target > 0:
                    self.btn_rec.config(text=f"Stop ({rec_time:.1f}s / {target:.1f}s)", bg='#f88')
                else:
                    self.btn_rec.config(text=f"Stop ({rec_time:.1f}s)", bg='#f88')
            else:
                self.recording_state = False
                self.btn_rec.config(text="Start Recording", bg='#ddd')
        
        # --- [新增] 更新进度条逻辑 ---
        if self.config.connection_mode == 'PLAYBACK':
            # 更新进度条数值 (0-100)
            self.pb_bar['value'] = pb_prog
            # 更新百分比文字显示
            self.lbl_prog_text.config(text=f"Progress: {pb_prog:.1f}%")
    # ----------------------------

class App:
    def __init__(self, root):
        self.root = root; self.root.geometry("1300x850"); self.config = RadarConfig(); self.config.load_layout("2x4_Default")
        if not os.path.exists(self.config.data_save_dir): os.makedirs(self.config.data_save_dir)
        self.data_manager = RadarDataManager(self.config); self.algo_processor = AlgorithmProcessor(self.config, self.data_manager)
        self.source = None; self.running = False
        self.root.columnconfigure(1, weight=1); self.root.rowconfigure(0, weight=1)
        cbs = {'start': self.start, 'stop': self.stop, 'rec_start': self.rec_start, 'rec_stop': self.rec_stop, 'update_layout': self.update_layout}
        self.ctrl = ControlPanel(root, self.config, cbs); self.ctrl.grid(row=0, column=0, sticky='ns')
        self.plot_panel = PlotPanel(root); self.plot_panel.grid(row=0, column=1, sticky='nsew')
        self.plot_panel.init_layout("PLOT", self.config.algo_params['PLOT'])
        self.pc_export_data = []  # 新增：用于存储点云导出数据的列表
        self.playback_queue = [] # 新增：存放待处理的文件路径队列
        self.playback_skip_state = False  # False 表示处理，True 表示跳过
    def update_layout(self, mode):
        # 1. 更新绘图面板的布局
        self.plot_panel.init_layout(mode, self.config.algo_params.get(mode, {}))
        
        # 2. 【核心修复】强制算法处理器重新初始化参数
        # 这样当你修改了虚拟天线索引、频率范围等参数时，后端才会重新计算
        self.algo_processor.init_done = False 
        print(f"参数已更新，算法 {mode} 将重新初始化...")
    def start(self):
        RadarProtocol.update_protocol(self.config.ft_len)
        # 新增：如果是回放模式，重置导出缓存
        if self.config.connection_mode == 'PLAYBACK':
            # 如果配置中有文件列表，则初始化队列
            file_list = getattr(self.config, 'playback_file_list', [])
            if file_list:
                self.playback_queue = file_list.copy()
                # 弹出第一个文件进行播放
                self._start_next_file()
            else:
                messagebox.showwarning("Warning", "Please select bin files first.")
                return
        else:
            # 实时模式逻辑不变
            self.source = LiveRadarSource(self.config)
            if not self.source.start(): return
            self.algo_processor.init_done = False; self.running = True; self.loop()
    
    def _start_next_file(self):
        """内部方法：从队列中提取下一个文件并启动播放"""
        if not self.playback_queue:
            messagebox.showinfo("Batch Finished", "All files have been processed.")
            return
        # 获取下一个文件
        current_file = self.playback_queue.pop(0)
        self.config.playback_file = current_file
        
        # --- 【关键修复：重置数据核心状态】 ---
        # 1. 重新实例化数据管理器，清空所有通道的 deque 缓冲区和背景参考
        self.data_manager = RadarDataManager(self.config)
        
        # 2. 重新实例化算法处理器，并关联新的数据管理器
        self.algo_processor = AlgorithmProcessor(self.config, self.data_manager)
        
        # 3. 重置导出缓存和 UI
        self.pc_export_data = []
        self.plot_panel.pc_history = [] 
        # ------------------------------------
        
        print(f"开始处理 ({len(self.playback_queue)} 剩余): {os.path.basename(current_file)}")
        
        # 启动播放源
        self.source = FilePlaybackSource(self.config)
        if not self.source.start(): 
            print(f"跳过损坏文件: {current_file}")
            self._start_next_file()
            return
            
        self.algo_processor.init_done = False; self.running = True; self.loop()
    def stop(self): 
        self.running = False; 
        # --- 新增：保存 JSON 文件逻辑 ---
        if self.config.connection_mode == 'PLAYBACK' and self.config.export_pc_json:
            if self.pc_export_data:
                self.save_pc_to_json()
            else:
                print("未检测到点云数据，跳过导出。")
        self.source.stop() if self.source else None; 
        self.ctrl.update_ui(False, False, 0, 0, 0)


    # 新增：保存函数
    def save_pc_to_json(self):
        # 使用当前正在播放的文件名
        current_bin = self.config.playback_file 
        if not current_bin: return
            
        json_path = os.path.splitext(current_bin)[0] + ".json" 
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                # 记得使用上个回答提到的 NpEncoder
                json.dump({
                    "source_file": current_bin,
                    "export_time": str(datetime.now()),
                    "total_frames": len(self.pc_export_data),
                    "data": self.pc_export_data
                }, f, indent=4) 
            print(f"✅ 自动导出成功: {os.path.basename(json_path)}")
            # 批量模式下建议取消弹窗，改用 print，否则会中断自动化流程
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    def rec_start(self, p, d): return self.source.start_recording(p, d)
    def rec_stop(self): self.source.stop_recording()
    def loop(self):
        if not self.running: return
        
        # 1. 获取新帧
        new_frames = self.source.get_batch_frames()
        
        # 2. 【核心修复】只有当收到新数据时，才进行处理和导出
        if new_frames:
            # 获取当前配置中的第一个天线对，作为一轮的起点标记
            first_pair = (self.config.udp_tx_list[0], self.config.udp_rx_list[0])

            for f in new_frames: 
                tx, rx, _ = f
                # 检测是否进入了新的一轮
                if (tx, rx) == first_pair:
                    # 仅在 PLAYBACK 模式且开关打开时触发跳帧逻辑
                    if self.config.connection_mode == 'PLAYBACK' and getattr(self.config, 'downsample_pb', False):
                        self.playback_skip_state = not self.playback_skip_state
                        # print(f"跳帧切换: {self.playback_skip_state}")
                    else:
                        # 实时模式或开关关闭时，始终不跳帧
                        self.playback_skip_state = False
                # 根据状态决定是否送入 data_manager
                if not self.playback_skip_state:
                    self.data_manager.process_frame(*f)
            
            # 只有在新帧触发了缓冲区更新后，才执行算法
            if self.data_manager.buffer_full:
                m = self.config.current_algo
                d = None
                
                if m == 'PLOT':
                    d = self.algo_processor.step_waveform()
                elif m == '2D-MUSIC':
                    d = self.algo_processor.step_2d_music()
                elif m == 'POINT-CLOUD':
                    d = self.algo_processor.step_point_cloud()
                    
                    # 导出逻辑：现在它只会在有新帧进来时运行一次
                    if d and self.config.connection_mode == 'PLAYBACK':
                        d['filename'] = os.path.basename(self.config.playback_file)
                        if self.config.export_pc_json:
                            export_frame = {
                                "frame_index": len(self.pc_export_data),
                                "breath_val": d.get("breath_val", 0.0),
                                "points": d.get("detected_points", [])
                            }
                            self.pc_export_data.append(export_frame)
                
                if d:
                    self.plot_panel.update_data(m, d)

        # 3. 【自动停止与批量衔接逻辑】
        if self.config.connection_mode == 'PLAYBACK':
            # 检查 source 线程是否已结束 (running=False) 且数据队列已取完
            if not self.source.running and self.source.data_queue.empty():
                # --- 当前文件结束 ---
                self.running = False

                # 1. 自动执行保存
                if self.config.export_pc_json and self.pc_export_data:
                    self.save_pc_to_json()
                
                # 2. 停止当前 source 释放资源
                if self.source: self.source.stop()
                # 3. 检查队列，尝试播放下一个
                if self.playback_queue:
                    # 使用 after 延迟一下，防止 UI 线程太拥挤
                    self.root.after(500, self._start_next_file)
                else:
                    print("批量任务全部完成")
                    self.stop() # 队列全空后，执行最后的清理
                return

        # 更新 UI 状态
        fps = getattr(self.source, 'measured_fps', getattr(self.source, 'playback_fps', 0))
        rec, t = self.source.get_rec_status()
        prog = self.source.get_progress() if hasattr(self.source, 'get_progress') else 0
        self.ctrl.update_ui(True, rec, t, fps, prog)
        
        self.root.after(20, self.loop)

if __name__ == '__main__':
    root = tk.Tk(); app = App(root); root.protocol("WM_DELETE_WINDOW", lambda: app.stop() or root.destroy()); root.mainloop()