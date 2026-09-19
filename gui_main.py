import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time
import json
import threading
import queue
import serial
import os
import subprocess
import struct
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from scipy import ndimage, signal
from scipy.signal.windows import chebwin
from device_management import *
from CAN_data_listen import CANFrameServer, radar_hardware_init
# --- External Modules ---
from UDP_data_listen import UDPFrameServer
from frame_sync import FrameSyncBuffer   # 串口字节流分帧器
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
# 0. Config Persistence
# ==============================================================================
class NpEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 序列化"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_save.json')

_SAVE_FIELDS = [
    'ft_len', 'max_snapshots', 'udp_tx_list', 'udp_rx_list', 'bg_m_factor',
    'num_tx_antennas', 'num_rx_antennas', 'rx_antenna_start_num',
    'connection_mode', 'serial_port', 'baud_rate', 'udp_ip', 'udp_port',
    'current_layout_name', 'current_algo', 'snapshot_rate', 'tap_interval_si',
    'data_save_dir', 'record_filename', 'record_duration',
    'playback_duration', 'export_pc_json', 'recent_save_dirs',
    'can_device_type', 'mcu_name', 'can_packet_size', 'can_header_size',
    'can_cir_data_size', 'can_uci_signature', 'can_packet_cnt',
]

def save_config(config):
    """将当前配置持久化到 JSON 文件"""
    try:
        data = {
            "version": 1,
            "radar_config": {k: getattr(config, k) for k in _SAVE_FIELDS if hasattr(config, k)},
            "algo_params": config.algo_params,
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=NpEncoder, ensure_ascii=False)
    except Exception as e:
        print(f"[Config] 保存配置失败: {e}")

def load_config(config):
    """从 JSON 文件加载配置, 覆盖默认值. 返回 True 表示加载成功."""
    if not os.path.exists(CONFIG_FILE):
        return False
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 顶层配置: 只覆盖 JSON 中存在的 key
        rc = data.get("radar_config", {})
        for key, val in rc.items():
            if hasattr(config, key):
                setattr(config, key, val)
        if config.connection_mode == "BD":
            config.connection_mode = "BD_UDP"
        # 算法参数: 浅合并, 保留代码默认值中的新 key
        saved_params = data.get("algo_params", {})
        for algo_name, params in saved_params.items():
            if algo_name in config.algo_params:
                config.algo_params[algo_name].update(params)
            else:
                config.algo_params[algo_name] = params
        # 确保天线布局与 current_layout_name 一致
        if 'RA-OCCUPANCY' in config.algo_params and 'range_bin_keep_range' in config.algo_params['RA-OCCUPANCY']:
            config.algo_params['RA-OCCUPANCY'].pop('range_bin_drop_front', None)
        config.load_layout(config.current_layout_name)
        print(f"[Config] 配置已从 {CONFIG_FILE} 加载")
        return True
    except Exception as e:
        print(f"[Config] 加载配置失败: {e}")
        return False

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
# 1. Presets & Defaults
# ==============================================================================
ANTENNA_LAYOUTS = {
    "2x4_Default": {
        "tx": [[0.0188, 0, 0.0], [0.0188, 0, 0.0]],
        "rx": [[0.0188*3, 0, 0.0188*2], [0.0188*2, 0, 0.0188*2], [0.0188, 0.0, 0.0188*2], [0.0, 0.0, 0.0188*2]],
        "tx_list": [2], "rx_list": [4, 5, 6, 7],
        "tx_num": 1, "rx_num": 4, "rx_start": 4,
        "virt_ant_x": [0.0, 0.0188, 0.0376, 0.0564],
        "indices_azimuth": [0, 1, 2, 3]
    },
    "4x4_Demo": {
        "tx": [[-0.03,0,0], [-0.01,0,0], [0.01,0,0], [0.03,0,0]],
        "rx": [[-0.03,0,0], [-0.01,0,0], [0.01,0,0], [0.03,0,0]],
        "tx_list": [1, 2, 3, 4], "rx_list": [5, 6, 7, 8],
        "tx_num": 4, "rx_num": 4, "rx_start": 5
    },
    "2x2_yzl": {
        "tx": [[0.0188*2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "rx": [[0.0188*2, 0.0, 0.0188], [0.0188, 0.0, 0.0188]],
        "tx_list": [1, 2], "rx_list": [1, 2],
        "tx_num": 2, "rx_num": 2, "rx_start": 1,
        "virt_ant_x": [0.0752, 0.0564, 0.0376, 0.0188],
        "indices_azimuth": [0, 1, 2, 3]
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
        "capon_diag_load": 1e-3,       # Capon 对角加载因子
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
        "aoa_calib_mode": "linear",     # 角度校准模式: 'linear' / 'dbf_only' / 'both'
        "aoa_offset_deg": -12.0,    # 角度偏移（度），正值代表向右偏
        "aoa_scale": 1.0,         # 角度缩放比例，默认为 1.0
        "ant_calib_en": False,        # 通道相位校准使能 (Capon/DBF 共用)
        "ant_calib_phase": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 8通道校准相位(弧度)
        "doppler_tx": 1,
        "doppler_rx": 6,
    },

    "POINT-CLOUD-OPTIMIZED": {
        "center_freq": 7.9872e9,
        "snapshots": 64,
        "cir_combine_num": 1,
        "leakage_offset": 5,
        "doppler_window": "chebyshev",
        "doppler_win_atten": 60,
        "doppler_dc_remove": True,
        "cfar_method": "ca-cfar",
        "cfar_threshold_db": 10.0,
        "range_train": 3, "range_guard": 1,
        "doppler_train": 4, "doppler_guard": 2,
        "cfar_velocity_min": 1,
        "cfar_velocity_max": 20,
        "cfar_range_min": 2,
        "cfar_range_max": 20,
        "cfar_range_peak_flag": False,
        "cfar_doppler_peak_flag": True,
        "subbin_refine_en": True,
        "aoa_method": "MUSIC",
        "fft_n": 16, "antenna_spacing": 0.5,
        "indices_azimuth": [2, 3, 6, 7],
        "music_forward_backward": False,
        "capon_diag_load": 1e-3,
        "ant_dbf_select": [2, 3, 6, 7],
        "azi_angle_range": [-90, 90],
        "azimuth_num": 64,
        "dbf_diff": 0,
        "aoa_calib_mode": "linear",
        "ant_calib_en": False,
        "ant_calib_phase": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # ---- 角度维 CFAR 寻峰 (替代 prominence, 局部噪声估计) ----
        "aoa_cfar_en": True,             # 启用角度CFAR寻峰
        "aoa_cfar_guard": 4,             # 保护单元半窗 (bin数)
        "aoa_cfar_train": 8,             # 训练单元半窗 (bin数)
        "aoa_cfar_threshold_db": 10.0,   # CFAR阈值 (dB)
        "aoa_debug_print": False,        # 调试打印: 角度谱/相位/寻峰详情
        "dist_per_tap": 0.1875,
        "point_lifetime_sec": 1.5,
        "plot_xlim": 1.5,
        "plot_ylim_min": -2.5, "plot_ylim_max": 0.0,
        "aoa_offset_deg": -12.0, "aoa_scale": 1.0,
        "doppler_tx": 1, "doppler_rx": 6,
        "cfar_only_selected": True,
    },

    "POINT-CLOUD-DUBHE": {
        "center_freq": 7.9872e9,
        "cir_combine_num": 5,
        "ring_buffer_len": 50,
        "slide_step": 20,
        "leakage_offset": 5,
        "doppler_win_en": True,
        "doppler_win_coef": 60,
        "doppler_dc_en": True,
        "doppler_fft": 64,
        "siso_ch": [2, 3, 6, 7],
        "cfar_low_r_idx": 2, "cfar_high_r_idx": 12,
        "cfar_low_v_idx": 1, "cfar_high_v_idx": 20,
        "cfar_th": [10, 10],
        "cfar_range_peak_flag": True,
        "cfar_doppler_peak_flag": True,
        "dynamic_filter_en": True,
        "noi_edges": [-15, -5, 10, 20],
        "cfar_th_dynamic": [15, 12, 10, 8, 7],
        "ant_dbf_select": [2, 3, 6, 7],
        "azi_angle_range": [-90, 90],
        "azimuth_num": 64,
        "dbf_diff": 0,
        "aoa_method": "DBF",
        "fft_n": 16,
        "antenna_spacing": 0.5,
        "music_forward_backward": False,
        "capon_diag_load": 1e-3,
        "aoa_calib_mode": "linear",
        "aoa_offset_deg": -12.0,
        "aoa_scale": 1.0,
        "ant_calib_en": False,
        "ant_calib_phase": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dist_per_tap": 0.1875,
        "point_lifetime_sec": 1.5,
        "plot_xlim": 1.5,
        "plot_ylim_min": -2.5, "plot_ylim_max": 0.0,
        "doppler_tx": 1, "doppler_rx": 6,
    },

    "POINT-CLOUD-PAPER": {
        # --- Physical ---
        "center_freq": 7.9872e9,
        "bandwidth": 100e6,
        "dist_per_tap": 0.1875,
        "cir_offset": 8,
        "snapshots": 64,

        # --- Antenna ---
        "num_virtual_ant": 8,
        "antenna_spacing": 0.5,
        "indices_azimuth": [2, 3, 6, 7],
        "cfar_only_selected": True,

        # --- Doppler preprocessing ---
        "doppler_window": "chebyshev",
        "doppler_win_atten": 60,
        "doppler_dc_remove": True,
        "leakage_offset": 5,

        # --- Pass 1: 1D Range CFAR (min(L,R) noise estimator) ---
        "range_train": 4,
        "range_guard": 2,
        "range_threshold_db": 10.0,
        "range_ave_pad": 3,
        "doppler_sum_v_min": 1,
        "doppler_sum_v_max": 20,

        # --- Pass 2: Angle quality check (替代 1D Angle CFAR) ---
        "angle_threshold_db": 8.0,
        "aoa_coarse_n": 36,
        "aoa_angle_range": [-70, 70],

        # --- AoA method (MUSIC / DBF / FFT) ---
        "aoa_method": "MUSIC",
        "fft_n": 16,
        "music_forward_backward": False,
        "capon_diag_load": 1e-3,

        # --- Pass 3: Zoom-in refinement ---
        "zoom_in_factor": 3,
        "zoom_threshold_gamma": 0.5,

        # --- DBF specific (shared with other modes) ---
        "ant_dbf_select": [2, 3, 6, 7],
        "azimuth_num": 64,
        "dbf_diff": 0,
        "ant_calib_en": False,
        "ant_calib_phase": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        # --- Calibration ---
        "aoa_calib_mode": "linear",
        "aoa_offset_deg": -12.0,
        "aoa_scale": 1.0,

        # --- Post-processing ---
        "point_lifetime_sec": 1.5,
        "plot_xlim": 1.5,
        "plot_ylim_min": -2.5,
        "plot_ylim_max": 0.0,

        # --- Breathing ---
        "doppler_tx": 1,
        "doppler_rx": 6,
    },

    "RA-CFAR": {
        "center_freq": 7.9872e9,
        "snapshots": 64,
        "cir_combine_num": 1,
        "heatmap_start_snapshots": 16,   # [加速] 热力图早启动阈值(帧数); 设0=回到等整缓冲
        "leakage_offset": 5,
        "doppler_window": "chebyshev",
        "doppler_win_atten": 60,
        "doppler_dc_remove": True,
        "indices_azimuth": [2, 3, 6, 7],
        "ant_dbf_select": [2, 3, 6, 7],
        "azi_angle_range": [-90, 90],
        "azimuth_num": 64,
        "ant_calib_en": False,
        "ant_calib_phase": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "capon_diag_load": 1e-3,            # Capon 对角加载 (论文 Eq.11 的 β)
        # ---- CFAR 检测 ----
        "cfar_mode": "2d_ca",              # '2d_ca'=二维CA-CFAR / 'two_pass'=论文Algorithm1
        # -- 2D CA-CFAR 参数 --
        "range_train": 3, "range_guard": 1,
        "azimuth_train": 4, "azimuth_guard": 2,
        "cfar_threshold_db": 10.0,
        "cfar_range_peak_flag": False,
        "cfar_azimuth_peak_flag": True,
        "cfar_range_min": 2, "cfar_range_max": 20,
        # -- Two-Pass CFAR 参数 (cfar_mode='two_pass' 时生效) --
        "pass1_N_W_R": 3, "pass1_N_G_R": 1, "pass1_N_ave_R": 2,
        "pass1_gamma_R_db": 10.0,
        "pass2_N_W_phi": 4, "pass2_N_G_phi": 2,
        "pass2_gamma_phi_db": 10.0,
        # ---- 点云参数 ----
        "dist_per_tap": 0.1875,
        "point_lifetime_sec": 1.5,
        "plot_xlim": 1.5,
        "plot_ylim_min": -2.5,
        "plot_ylim_max": 0.0,
        "aoa_offset_deg": -12.0,
        "aoa_scale": 1.0,
        "aoa_calib_mode": "linear",
        "aoa_debug_print": False,
    },

    "RA-OCCUPANCY": {
        # ---- 热力图计算 (与 RA-CFAR 共用 _compute_ra_heatmap) ----
        "center_freq": 7.9872e9,
        "snapshots": 64,
        "cir_combine_num": 1,
        "heatmap_start_snapshots": 64,   # [加速] 占用流水线攒够一个窗口(64)即开跑, 不必等540
        "leakage_offset": 5,
        "range_bin_keep_range": [5, 13],
        "range_zero_bin": 5.0,          # 原始 CIR 中对应物理 0 m 的 range bin
        "doppler_window": "chebyshev",
        "doppler_win_atten": 60,
        "doppler_dc_remove": True,
        "indices_azimuth": [2, 3, 6, 7],
        "ant_dbf_select": [2, 3, 6, 7],
        "azi_angle_range": [-90, 90],
        "azimuth_num": 64,
        "ant_calib_en": False,
        "ant_calib_phase": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "capon_diag_load": 1e-3,
        "dist_per_tap": 0.1875,
        # ---- 后处理 ----
        "smooth_kernel": [2, 4],        # 卷积平滑核大小 [rows, cols], 可调
        "oa_mean_threshold": 0.0,
        "sample_enable": True,          # True=enable TFLite model, False=bypass model
        "oa_model_path": "./model/epoch-25-val-f1-100.0-sp-100.0.tflite",
        "oa_filter_window_size": 5,
        "oa_filter_in_threshold": 3,
        "playback_sample_export": 0,    # 1=回放时导出标准化后的网络输入样本
        # ---- 可视化 ----
        "heatmap_update_stride_combined": 4,
        "plot_xlim": 1.5,
        "plot_ylim_min": -2.5,
        "plot_ylim_max": -0.1,
        "heatmap_clim_mode": "auto",
        "heatmap_clim_vmin": -80,
        "heatmap_clim_vmax": 0,
        "heatmap_background_color": "#f2f2f2",
        # ---- 显示模式: 热力图 / 定位图(仅峰值红点) ----
        "ra_occ_show_localization": False,      # False=热力图(默认) / True=定位图
        "ra_occ_marker_size": 8.0,              # 定位图红点大小 (pt)
        "ra_occ_localization_smooth": "mode",   # off/mode/mean: 红点位置取最近N帧众数/均值
        "ra_occ_localization_hold_sec": 1.0,    # >0: 峰值失效后红点保持该秒数再消失
    },

    "SEAT-OCCUPANCY": {
        "enable": True,
        "seat_type": "4_seats",
        "seats_4": [
            {"name": "1", "ra_peak_ratio": 0.3,
             "main": {"cx": -0.30, "cy": -0.60, "rx": 0.20, "ry": 0.20},
             "adult_threshold": 0.02,
             "child_threshold_low": 0.05, "child_threshold_high": 5.0,
             "child_special": {"cx": -0.25, "cy": -0.70, "rx": 0.15, "ry": 0.15,
                               "threshold_low": 0.05, "threshold_high": 5.0}},
            {"name": "2", "ra_peak_ratio": 0.3,
             "main": {"cx":  0.30, "cy": -0.60, "rx": 0.20, "ry": 0.20},
             "adult_threshold": 0.02,
             "child_threshold_low": 0.05, "child_threshold_high": 5.0,
             "child_special": {"cx":  0.25, "cy": -0.70, "rx": 0.15, "ry": 0.15,
                               "threshold_low": 0.05, "threshold_high": 5.0}},
            {"name": "3", "ra_peak_ratio": 0.3,
             "main": {"cx": -0.30, "cy": -1.40, "rx": 0.20, "ry": 0.20},
             "adult_threshold": 0.02,
             "child_threshold_low": 0.05, "child_threshold_high": 5.0,
             "child_special": {"cx": -0.25, "cy": -1.50, "rx": 0.15, "ry": 0.15,
                               "threshold_low": 0.05, "threshold_high": 5.0}},
            {"name": "4", "ra_peak_ratio": 0.3,
             "main": {"cx":  0.30, "cy": -1.40, "rx": 0.20, "ry": 0.20},
             "adult_threshold": 0.02,
             "child_threshold_low": 0.05, "child_threshold_high": 5.0,
             "child_special": {"cx":  0.25, "cy": -1.50, "rx": 0.15, "ry": 0.15,
                               "threshold_low": 0.05, "threshold_high": 5.0}},
        ],
        "seats_5": [
            {"name": "1", "ra_peak_ratio": 0.3,
             "main": {"cx": -0.30, "cy": -0.50, "rx": 0.25, "ry": 0.20},
             "adult_threshold": 0.02,
             "child_threshold_low": 0.05, "child_threshold_high": 5.0,
             "child_special": {"cx": -0.25, "cy": -0.60, "rx": 0.19, "ry": 0.15,
                               "threshold_low": 0.05, "threshold_high": 5.0}},
            {"name": "2", "ra_peak_ratio": 0.3,
             "main": {"cx":  0.30, "cy": -0.50, "rx": 0.25, "ry": 0.20},
             "adult_threshold": 0.02,
             "child_threshold_low": 0.05, "child_threshold_high": 5.0,
             "child_special": {"cx":  0.25, "cy": -0.60, "rx": 0.19, "ry": 0.15,
                               "threshold_low": 0.05, "threshold_high": 5.0}},
            {"name": "3", "ra_peak_ratio": 0.3,
             "main": {"cx": -0.40, "cy": -1.30, "rx": 0.17, "ry": 0.20},
             "adult_threshold": 0.02,
             "child_threshold_low": 0.05, "child_threshold_high": 5.0,
             "child_special": {"cx": -0.35, "cy": -1.40, "rx": 0.13, "ry": 0.15,
                               "threshold_low": 0.05, "threshold_high": 5.0}},
            {"name": "4", "ra_peak_ratio": 0.3,
             "main": {"cx":  0.40, "cy": -1.30, "rx": 0.17, "ry": 0.20},
             "adult_threshold": 0.02,
             "child_threshold_low": 0.05, "child_threshold_high": 5.0,
             "child_special": {"cx":  0.35, "cy": -1.40, "rx": 0.13, "ry": 0.15,
                               "threshold_low": 0.05, "threshold_high": 5.0}},
            {"name": "5", "ra_peak_ratio": 0.3,
             "main": {"cx":  0.00, "cy": -1.30, "rx": 0.17, "ry": 0.20},
             "adult_threshold": 0.02,
             "child_threshold_low": 0.05, "child_threshold_high": 5.0,
             "child_special": {"cx":  0.00, "cy": -1.40, "rx": 0.13, "ry": 0.15,
                               "threshold_low": 0.05, "threshold_high": 5.0}},
        ],
        "smooth_window": 3,
        "smooth_threshold": 0.5,
        "hold_time_sec": 2.0,
    },


}

# ==============================================================================
# 2. Protocol & Config
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
        tx = frame_bytes[RadarProtocol.HEADER_LEN];
        rx = frame_bytes[RadarProtocol.HEADER_LEN+1]
        cir = frame_bytes[RadarProtocol.HEADER_LEN+2 : -RadarProtocol.FOOTER_LEN]
        s16 = np.frombuffer(cir, dtype=np.int16)
        c_data = s16[0::2].astype(np.float32) + 1j * s16[1::2].astype(np.float32)
        return tx, rx, c_data

class ConfigAdapter:
    def __init__(self, dynamic_config):
        # UDP 字段
        self.UDP_IP = dynamic_config.udp_ip; self.UDP_PORT = dynamic_config.udp_port
        self.UDP_TX_LIST = dynamic_config.udp_tx_list; self.UDP_RX_LIST = dynamic_config.udp_rx_list
        self.FT_LEN = dynamic_config.ft_len
        self.START_SIGN = RadarProtocol.START_SIGN; self.STOP_SIGN = RadarProtocol.STOP_SIGN
        # CAN 字段 (CANFrameServer 使用不同命名，桥接过来)
        self.TX_LIST = dynamic_config.udp_tx_list
        self.RX_LIST = dynamic_config.udp_rx_list
        self.MCU_NAME = dynamic_config.mcu_name
        self.PACKET_SIZE = dynamic_config.can_packet_size
        self.HEADER_SIZE = dynamic_config.can_header_size
        self.CIR_DATA_SIZE = dynamic_config.can_cir_data_size
        self.UCI_SIGNATURE = bytes.fromhex(dynamic_config.can_uci_signature)
        self.PACKET_CNT = dynamic_config.can_packet_cnt

@dataclass
class RadarConfig:
    # --- 硬件与通信 ---
    ft_len: int = 32
    max_snapshots: int = 540
    udp_tx_list: List[int] = field(default_factory=lambda: [1, 2])
    udp_rx_list: List[int] = field(default_factory=lambda: [4, 5, 6, 7])
    bg_m_factor: float = 4
    num_tx_antennas: int = 2
    num_rx_antennas: int = 4
    rx_antenna_start_num: int = 4
    
    connection_mode: str = "UDP"  # "UDP", "BD_UDP", "BD_CAN", "SERIAL", "CAN", "PLAYBACK"
    serial_port: str = "COM6"
    baud_rate: int = 460800
    udp_ip: str = "127.0.0.1"
    udp_port: int = 55555

    # CAN 通信参数
    can_device_type: str = "ZLGCAN"       # "ZLGCAN" 或 "TOOMOSS"
    mcu_name: str = "Calterah"            # "Calterah" 或 "29D6"
    can_packet_size: int = 320            # 完整传输单元大小 (多个 CAN 帧拼成)
    can_header_size: int = 20             # UCI(4B) + Payload Header(16B)
    can_cir_data_size: int = 256          # CIR 数据区字节数
    can_uci_signature: str = "0f001001"   # UCI 同步头 (hex 字符串, 不含空格)
    can_packet_cnt: int = 4               # 每快照的分块数

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
    recent_save_dirs: List[str] = field(default_factory=lambda: ["./data"])
    virt_ant_x: Optional[List[float]] = None       # [D4] 虚拟天线 x 坐标（随布局加载）
    layout_indices: Optional[List[int]] = None     # [D4] 方位角通道索引（随布局加载）
    def load_layout(self, name):
        if name in ANTENNA_LAYOUTS:
            self.current_layout_name = name
            L = ANTENNA_LAYOUTS[name]
            self.tx_positions = L['tx']; self.rx_positions = L['rx']
            self.udp_tx_list = L['tx_list']; self.udp_rx_list = L['rx_list']
            self.num_tx_antennas = L['tx_num']; self.num_rx_antennas = L['rx_num']
            self.rx_antenna_start_num = L['rx_start']
            self.virt_ant_x = L.get('virt_ant_x', None)
            self.layout_indices = L.get('indices_azimuth', None)

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
# 3. Data Manager
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

    def snapshot_count(self):
        """当前各通道已缓存快照数的最小值（按最短通道，防对齐错位）。"""
        lens = [len(b['complex']) for b in self.snapshots_data.values()]
        return min(lens) if lens else 0

    def ready_for(self, n_snapshots):
        """是否已有至少 n_snapshots 个可用快照（不必等整缓冲填满）。"""
        return self.buffer_full or self.snapshot_count() >= n_snapshots

    def process_frame(self, tx, rx, raw):
        if (tx, rx) not in self.snapshots_data: return None
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

        return r_no_bg, r_abs_no_bg

    def get_all_snapshot_as_array(self, type='complex', min_snapshots=None):
        """取整窗数据 (num_rx, num_tx, ft_len, L)。

        min_snapshots=None : 旧行为——必须整缓冲满(buffer_full)才可取；
        min_snapshots=N    : 缓冲未满但已有 >=N 帧时也返回（供 RA 热力图早启动），
                             取实际可用长度对齐。
        """
        if min_snapshots is not None:
            if self.snapshot_count() < min_snapshots:
                return None
        elif not self.buffer_full:
            return None
        
        # 1. 确定当前所有缓存中的最小有效长度
        # 这是为了防止某些通道只有 126 帧，而另一些有 127 帧导致的广播错误
        all_buffs = [b[type] for b in self.snapshots_data.values()]
        actual_lens = [len(b) for b in all_buffs]
        min_len = min(actual_lens) if actual_lens else 0
        
        # 2. 统一对齐到最小有效长度，防止广播错误
        # 实时模式下各通道到达时序不同（串口逐通道到达）或偶发丢帧时，
        # 通道间帧数会不一致；回放模式下文件尾部也可能不足量。
        # 取 min_len 保证 arr 最后一维与实际可用帧数匹配。
        target_len = self.config.max_snapshots
        if min_len < target_len:
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
        # 虚拟天线 x 坐标 (m)：优先用布局配置（随板子走），否则默认 8 通道
        virt_x = getattr(self.config, 'virt_ant_x', None)
        if virt_x is None:
            virt_x = np.array([-0.038, 0.0, -0.038, -0.019, 0.0, 0.038, 0.0, 0.019])
        else:
            virt_x = np.array(virt_x)
        channels = getattr(self.config, 'layout_indices', None)
        if channels is None:
            channels = params.get('indices_azimuth', list(range(len(virt_x))))
        ant_x = virt_x[channels] / wavelength  # 归一化到波长
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
            if getattr(self, '_dbf_sv_cache', None) is None:
                wavelength = 2.99792458e8 / params['center_freq']
                channels = params.get('ant_dbf_select', valid_indices)
                all_virt_x = np.array([-0.038, 0.0, -0.038, -0.019,
                                       0.0, 0.038, 0.0, 0.019])
                ant_x = all_virt_x[channels] / wavelength
                self._dbf_angles_cache = np.linspace(
                    np.deg2rad(angle_deg_range[0]),
                    np.deg2rad(angle_deg_range[1]),
                    params.get('azimuth_num', 64))
                self._dbf_sv_cache = np.exp(
                    -1j * 2 * np.pi *
                    ant_x[:, np.newaxis] * np.sin(self._dbf_angles_cache[np.newaxis, :]))
                if params.get('ant_calib_en', False):
                    calib_phase = np.array(params.get('ant_calib_phase',
                                          [0.0] * 8), dtype=np.float64)
                    calib = np.exp(1j * calib_phase[channels])
                    self._dbf_sv_cache = self._dbf_sv_cache * calib[:, np.newaxis]

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
        all_virt_x = np.array([0.0 * wavelength , 0.5*wavelength , 1*wavelength, 0.5*wavelength, 0.5*wavelength, wavelength, 0*wavelength, -0.5*wavelength])
        ant_x = all_virt_x[channels] / wavelength

        azi_deg = np.linspace(params['azi_angle_range'][0],
                              params['azi_angle_range'][1],
                              params['azimuth_num'])
        self._dbf_angles = np.deg2rad(azi_deg)
        self._dbf_sv = np.exp(-1j * 2 * np.pi *
                              ant_x[:, np.newaxis] * np.sin(self._dbf_angles[np.newaxis, :]))

        # 相位标定: 对齐 Dubhe 文档 Section 3.8.3 / Table 9
        if params.get('ant_calib_en', False):
            calib_phase = np.array(params.get('ant_calib_phase',
                                  [0] * 8), dtype=np.float64)
            calib = np.exp(1j * calib_phase[channels])
            self._dbf_sv = self._dbf_sv * calib[:, np.newaxis]

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
        print("[CAPON] channels=", channels, "ant_x(λ)=", np.round(ant_x, 3),
              "calib_en=", params.get('ant_calib_en'))

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
        N_snaps = params['snapshots']
        # [加速] 不必等整缓冲(默认540)填满: 攒够 heatmap_start_snapshots(默认≤窗口)即出图
        start_min = max(4, int(params.get('heatmap_start_snapshots', min(N_snaps, 16))))
        all_c = self.dm.get_all_snapshot_as_array('complex', min_snapshots=start_min)
        if all_c is None:
            return None, None, None

        cir_comb = params.get('cir_combine_num', 1)
        leakage_offset = params['leakage_offset']
        current_cube = all_c[:, :, :, -N_snaps:]

        num_rx = current_cube.shape[0]
        num_tx = current_cube.shape[1]
        num_chan = num_rx * num_tx

        # 预处理
        if cir_comb > 1:
            n_comb = current_cube.shape[3] // cir_comb
            trim = n_comb * cir_comb
            current_cube = current_cube[:, :, :16, :trim] \
                .reshape(num_rx, num_tx, 16, n_comb, cir_comb).mean(axis=4)
        current_cube, range_bin_start = apply_range_bin_selection(
            current_cube, params, return_start_bin=True)

        # 展平全部通道 → 按 indices_azimuth 选取
        # current_cube: (num_rx, num_tx, 32 range, L chirps)
        cube_flat = current_cube.transpose(1, 0, 2, 3).reshape(
            num_chan, current_cube.shape[2], current_cube.shape[3])
        valid_indices = getattr(self.config, 'layout_indices', None)
        if valid_indices is None:
            valid_indices = params.get('indices_azimuth',
                                       list(range(num_chan)))
        A_all = cube_flat[valid_indices, :, :]  # (len(valid_indices), 32, L)

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
            R_k += np.eye(R_k.shape[0]) * diag_load * np.abs(np.trace(R_k))
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
# ==============================================================================
class SeatOccupancyDetector:
    """
    基于规则的座椅占用检测器, 对齐论文公式 (12)-(16).
    输入任意点云算法的 detected_points, 输出每座椅占用状态.
    """

    def __init__(self, params):
        self.params = params
        self._load_seats()
        self.state_history = {s['name']: deque(maxlen=params.get('smooth_window', 3))
                              for s in self.seats}
        # 当前平滑后的占用状态
        self.occupancy = {s['name']: 0 for s in self.seats}
        # 最近一次的 f_k 值, 用于调试/显示
        self.f_values = {s['name']: 0.0 for s in self.seats}
        # 占用状态保持: 记录每个座椅的 hold 截止时间戳
        self.hold_until = {s['name']: 0.0 for s in self.seats}

    def _load_seats(self):
        seat_type = self.params.get('seat_type', '4_seats')
        key = 'seats_4' if seat_type == '4_seats' else 'seats_5'
        self.seats = self.params.get(key, self.params.get('seats_4', []))

    def _point_in_ellipse(self, x, y, seat):
        """公式 (12): 判断点是否在椭圆区域内"""
        xn = (x - seat['cx']) / seat['rx']
        yn = (y - seat['cy']) / seat['ry']
        return (xn * xn + yn * yn) < 1.0

    def _compute_dispersion(self, points):
        """公式 (13): 计算点集的空间分散度 σ_k"""
        if len(points) < 2:
            return 0.0
        pts = np.array(points)
        mean = np.mean(pts, axis=0)
        dists = np.sqrt(np.sum((pts - mean) ** 2, axis=1))
        return float(np.sqrt(np.mean(dists ** 2)))

    def process(self, detected_points):
        """
        输入: detected_points = [{'pos': (x,y), ...}, ...]
        输出: occupancy dict {'1':0/1, '2':0/1, '3':0/1, '4':0/1}
        """
        N = len(detected_points)
        if N == 0:
            # 无点云 → 所有座椅平滑为 0
            for name in self.occupancy:
                self.state_history[name].append(0)
            self._smooth()
            return dict(self.occupancy), dict(self.f_values)

        # 1. 椭圆筛选 + 每座椅 N_k 和 σ_k
        seat_points = {}
        for seat in self.seats:
            ellipse = seat.get('adult', seat)  # 优先成人椭圆, fallback 兼容旧配置
            pts = [(p['pos'][0], p['pos'][1]) for p in detected_points
                   if self._point_in_ellipse(p['pos'][0], p['pos'][1], ellipse)]
            seat_points[seat['name']] = pts

        N_list = [len(seat_points[s['name']]) for s in self.seats]
        sigma_list = [self._compute_dispersion(seat_points[s['name']])
                      for s in self.seats]
        print(sigma_list)
        # 2. 公式 (14): 归一化特征 f_k
        products = [sigma_list[i] * (N_list[i] / N) for i in range(len(self.seats))]
        denom = sum(products)
        if denom < 1e-12:
            f_values_list = [0.0] * len(self.seats)
        else:
            f_values_list = [p / denom for p in products]
            
        for i, seat in enumerate(self.seats):
            self.f_values[seat['name']] = f_values_list[i]
            
        # 3. 公式 (15): 阈值判决
        for i, seat in enumerate(self.seats):
            so_instant = 1 if f_values_list[i] > seat.get('th', 0.01) else 0
            
            self.state_history[seat['name']].append(so_instant)

        # 4. 公式 (16): 滑动平均平滑
        self._smooth()
        return dict(self.occupancy), dict(self.f_values)

    def _smooth(self):
        """公式 (16): 滑动平均 + 阈值 m=0.5, 含占用保持 (hold_time_sec)"""
        m = self.params.get('smooth_threshold', 0.5)
        hold_t = self.params.get('hold_time_sec', 0.0)
        now = time.time()
        for name, hist in self.state_history.items():
            if len(hist) > 0:
                avg = sum(hist) / len(hist)
                self.occupancy[name] = 1 if avg > m else 0
            # 占用保持: 一旦检出有人, 在 hold_time_sec 内强制保持占用状态
            if self.occupancy[name] == 1:
                self.hold_until[name] = now + hold_t
            elif hold_t > 0 and now < self.hold_until[name]:
                self.occupancy[name] = 1

    def get_seat_ellipses(self):
        """供 PlotPanel 绘图使用, 返回椭圆参数列表"""
        return [{'center': (s.get('adult', s)['cx'], s.get('adult', s)['cy']),
                 'rx': s.get('adult', s)['rx'], 'ry': s.get('adult', s)['ry'],
                 'name': s['name']} for s in self.seats]


class RAOccupancyDetector:
    """
    RA热力图能量占位检测器 (无 CFAR / 无点云).

    新决策逻辑:
      主检测区 (main, 合并后 = 原 adult 椭圆几何):
        ADULT:  main_max > adult_threshold  AND  main_max >= ra_peak_ratio * max(所有座位 main_max) → state=2
        CHILD:  child_threshold_low < main_max < child_threshold_high (无 ratio 条件) → state=1
        否则 → state=0

      特判区 (child_special = 原 child 椭圆):
        spec_threshold_low < special_max < spec_threshold_high → 强制 CHILD (state=1)
        覆盖主检测区的 EMPTY, 但不会把 EMPTY 冲成 CHILD 后再被 ADULT 覆盖
        (ADULT 在主检测区已判出时优先级更高)
    """

    def __init__(self, params):
        self.params = params
        self._load_seats()
        win_size = params.get('smooth_window', 3)
        self.state_window = {s['name']: deque(maxlen=win_size) for s in self.seats}
        self.occupancy = {s['name']: 0 for s in self.seats}
        self.energy_values = {s['name']: {'main': 0.0, 'child_special': 0.0} for s in self.seats}
        self.hold_until = {s['name']: 0.0 for s in self.seats}
        self._last_state = {s['name']: 0 for s in self.seats}

    def _load_seats(self):
        seat_type = self.params.get('seat_type', '4_seats')
        key = 'seats_4' if seat_type == '4_seats' else 'seats_5'
        self.seats = self.params.get(key, self.params.get('seats_4', []))

    def process(self, energy_dict):
        """
        energy_dict: {'1': {'main': 0.052, 'child_special': 0.030}, ...}
        返回: (occupancy_dict, energy_dict, state_dict)
              state: 0=empty, 1=child, 2=adult
        """
        for s in self.seats:
            name = s['name']
            e = energy_dict.get(name, {})
            if isinstance(e, dict):
                self.energy_values[name] = {'main': e.get('main', 0.0),
                                            'child_special': e.get('child_special', 0.0)}
            else:
                self.energy_values[name] = {'main': float(e), 'child_special': 0.0}

        # 全局主检测区最大值 (用于 ra_peak_ratio 相对比较)
        max_main = max(self.energy_values[n]['main'] for n in self.energy_values) \
                   if self.energy_values else 0.0

        for seat in self.seats:
            name = seat['name']
            main_e = self.energy_values[name]['main']
            special_e = self.energy_values[name]['child_special']
            ratio = seat.get('ra_peak_ratio', 0.3)

            # ── 主检测区判决 ──
            # 成人: threshold + ra_peak_ratio 双条件
            adult_th = seat.get('adult_threshold', 0.02)
            if main_e > adult_th and (max_main == 0 or main_e >= ratio * max_main):
                self.state_window[name].append(2)  # ADULT
            else:
                # 娃娃: 区间判断 (无 ratio 条件)
                child_lo = seat.get('child_threshold_low', 0.05)
                child_hi = seat.get('child_threshold_high', 5.0)
                if child_lo < main_e < child_hi:
                    self.state_window[name].append(1)  # CHILD
                else:
                    self.state_window[name].append(0)  # EMPTY (暂定)

            # ── 特判区: 角坑兜底 ──
            # 仅当主检测区未检出成人时, 特判区才能覆盖为 CHILD
            special_cfg = seat.get('child_special', {})
            if special_cfg:
                spec_lo = special_cfg.get('threshold_low', 0.05)
                spec_hi = special_cfg.get('threshold_high', 5.0)
                if spec_lo < special_e < spec_hi:
                    # 只覆盖 0 (EMPTY) 或 1 (CHILD), 不覆盖已经判出的 ADULT
                    if self.state_window[name][-1] != 2:
                        self.state_window[name][-1] = 1  # 强制 CHILD

        self._smooth()

        # 计数平滑后: 若占用 → 窗口内非零帧全是 1 才是小孩, 否则成人
        state = {}
        for s in self.seats:
            name = s['name']
            if self.occupancy[name] == 0:
                state[name] = 0
            else:
                window = list(self.state_window[name])
                non_zero = [v for v in window if v > 0]
                if not non_zero:
                    state[name] = self._last_state.get(name, 0)
                elif all(v == 1 for v in non_zero):
                    state[name] = 1  # 全是小孩 → child
                    self._last_state[name] = 1
                else:
                    state[name] = 2  # 出现过成人 → adult
                    self._last_state[name] = 2

        return dict(self.occupancy), dict(self.energy_values), state

    def _smooth(self):
        """计数平滑: 窗口内非零帧数 >= min_count → 占用. 含 hold_time 保持."""
        win_size = self.params.get('smooth_window', 3)
        ratio = self.params.get('smooth_threshold', 0.5)
        min_count = max(1, int(np.ceil(win_size * ratio)))
        hold_t = self.params.get('hold_time_sec', 0.0)
        now = time.time()
        for name, window in self.state_window.items():
            w = list(window)
            occupied_frames = sum(1 for v in w if v > 0)
            if occupied_frames >= min_count:
                self.occupancy[name] = 1
            else:
                self.occupancy[name] = 0
            # hold_time 保持
            if self.occupancy[name] == 1:
                self.hold_until[name] = now + hold_t
            elif hold_t > 0 and now < self.hold_until[name]:
                self.occupancy[name] = 1

    def get_seat_ellipses(self):
        """供 PlotPanel 绘图, 使用主检测区 (main) 椭圆"""
        return [{'center': (s.get('main', s)['cx'], s.get('main', s)['cy']),
                 'rx': s.get('main', s)['rx'], 'ry': s.get('main', s)['ry'],
                 'name': s['name']} for s in self.seats]


# ==============================================================================
# 5. IO Layer
# ==============================================================================
class LiveRadarSource:
    def __init__(self, config: RadarConfig):
        self.config = config; self.running = False; self.thread = None
        self.data_queue = queue.Queue(maxsize=5000); self.udp_server = None; self.can_server = None
        self.is_recording = False; self.record_file = None; self.record_lock = threading.Lock()
        self.rec_start_time = 0; self.rec_duration_target = 0; self.measured_fps = 0.0; self.last_fps_time = time.time(); self.frame_count_sec = 0
        self._frame_sync = FrameSyncBuffer(RadarProtocol.START_SIGN, RadarProtocol.FRAME_LEN)  # 串口字节流分帧器
        self._serial_pending = []   # 一次读出多帧时，暂存多余的完整帧
    def start(self):
        self.running = True; RadarProtocol.update_protocol(self.config.ft_len)
        # 帧长随 ft_len 变化，重建分帧器确保帧长与协议一致
        self._frame_sync = FrameSyncBuffer(RadarProtocol.START_SIGN, RadarProtocol.FRAME_LEN)
        self._serial_pending = []
        if self.config.connection_mode in ('UDP', 'BD_UDP'):
            self.udp_server = UDPFrameServer(ConfigAdapter(self.config)); self.udp_server.start()
        elif self.config.connection_mode in ('CAN', 'BD_CAN'):

            ret = init_device(self.config.can_device_type)


            if "失败" in str(ret):
                print(f"[LiveRadarSource] CAN 设备初始化失败: {ret}")
                self.running = False; return False
            print(f"[LiveRadarSource] {ret}")
            radar_hardware_init()
            self.can_server = CANFrameServer(ConfigAdapter(self.config))
            self.can_server.start()
            print(f"[LiveRadarSource] {self.config.connection_mode} 接收模式已启动")
        self.thread = threading.Thread(target=self._io_loop, daemon=True); self.thread.start(); return True
    def stop(self):
        self.running = False; self.stop_recording()
        if self.thread: self.thread.join()
        if self.udp_server: self.udp_server.stop()
        if self.can_server:
            self.can_server.stop()
            from device_management import deinit_device
            deinit_device(self.config.can_device_type)
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
    def _pull_serial_frame(self, ser) -> bytes:
        """读串口字节流，通过 FrameSyncBuffer 返回恰好一帧（或 b''）。

        一次读出的多帧会暂存在 _serial_pending 中逐轮返回，不丢帧；
        chunk 为空时仍会消化分帧器缓冲中的积压帧。
        """
        if self._serial_pending:
            return self._serial_pending.pop(0)
        chunk = ser.read(ser.in_waiting) if ser.in_waiting else b''
        frames = self._frame_sync.feed(chunk)
        if len(frames) > 1:
            self._serial_pending.extend(frames[1:])
        return frames[0] if frames else b''

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
                if self.config.connection_mode in ('UDP', 'BD_UDP'): f = self.udp_server.get_frame(); raw_chunk = f if f else b''; time.sleep(0.002) if not f else None
                elif self.config.connection_mode in ('CAN', 'BD_CAN'): f = self.can_server.get_frame(); raw_chunk = f if f else b''; time.sleep(0.002) if not f else None
                elif self.config.connection_mode == 'SERIAL': raw_chunk = self._pull_serial_frame(ser); time.sleep(0.002) if not raw_chunk else None
            except: pass
            if not raw_chunk: continue
            if self.is_recording and self.record_file: self.record_file.write(raw_chunk)
            parsed = RadarProtocol.parse_frame(raw_chunk)
            if parsed:
                raw_frame = bytes(raw_chunk)
                self.frame_count_sec += 1; self.data_queue.put(parsed + (raw_frame,)) if not self.data_queue.full() else None
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
                        self.data_queue.put(parsed + (bytes(chunk),))
                
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
# 6. UI
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
                    s = val.strip().replace('\uff0c', ',')
                    if s.startswith('(') and s.endswith(')'):
                        s = '[' + s[1:-1] + ']'
                    elif not s.startswith('['):
                        s = '[' + s + ']'
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
        self.ra_occ_peak_history = deque(maxlen=5)
        self.ra_occ_oa_filter_history = deque(maxlen=5)
    def init_layout(self, mode, params):
        self.figure.clf(); self.axes = {}; self.plots = {}
        self.ra_occ_peak_history.clear()
        try:
            filter_window_size = max(1, int(params.get('oa_filter_window_size', 5)))
        except (TypeError, ValueError):
            filter_window_size = 5
        self.ra_occ_oa_filter_history = deque(maxlen=filter_window_size)
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
        elif mode in ('POINT-CLOUD', 'POINT-CLOUD-OPTIMIZED', 'POINT-CLOUD-DUBHE', 'POINT-CLOUD-PAPER', 'RA-CFAR'):
            ax = self.figure.add_subplot(111)
            ax.set_xlim(-params.get('plot_xlim', 1.5), params.get('plot_xlim', 1.5))
            ax.set_ylim(params.get('plot_ylim_min', -2.5), params.get('plot_ylim_max', 0))
            mode_titles = {
                'POINT-CLOUD': "Vehicle Occupancy Point Cloud (CA-CFAR)",
                'POINT-CLOUD-OPTIMIZED': "Vehicle Occupancy Point Cloud (Optimized)",
                'POINT-CLOUD-DUBHE': "Vehicle Occupancy Point Cloud (Dubhe CPD)",
                'POINT-CLOUD-PAPER': "Vehicle Occupancy Point Cloud (Multipass CFAR, IEEE JSEN'24)",
                'RA-CFAR': "Vehicle Occupancy (Range-Azimuth Two-Pass CFAR)",
            }
            ax.set_title(mode_titles.get(mode, "Point Cloud"))
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.grid(True, linestyle=':', alpha=0.6)
            self.plots['pc_scatter'] = ax.scatter([], [], c=[], cmap='cool', s=30, alpha=0.8)
            occ_params = params.copy()
            occ_params['occupancy_config'] = params.get('occupancy_config', None)
            self._draw_seating_ellipses(ax, occ_params)
            self.axes['main'] = ax
        elif mode == 'RA-HEATMAP':
            ax = self.figure.add_subplot(111)
            ax.set_title("Capon Range-Azimuth Heatmap")
            ax.set_xlabel("Azimuth (°)")
            ax.set_ylabel("Range (m)")
            self.plots['ra_hm'] = ax.imshow(
                np.zeros((32, 64)), aspect='auto', origin='lower',
                cmap='jet', interpolation='bilinear')
            self.axes['main'] = ax
        elif mode == 'RA-OCCUPANCY':
            gs = self.figure.add_gridspec(
                2, 2, width_ratios=[2, 1], height_ratios=[3, 1],
                hspace=0.08)
            ax_hm = self.figure.add_subplot(gs[0, 0])
            ax_info = self.figure.add_subplot(gs[1, 0])
            ax_occ = self.figure.add_subplot(gs[:, 1])

            # 左: Cartesian RA 热力图 + 座位椭圆
            # ax_hm.set_title("RA Occupancy Heatmap (Cartesian)")
            ax_hm.set_xlabel("X (m)")
            ax_hm.set_ylabel("Y (m)")
            ax_hm.grid(True, linestyle=':', alpha=0.5)
            heatmap_bg_color = params.get('heatmap_background_color', '#f2f2f2')
            heatmap_cmap = plt.get_cmap('jet').copy()
            heatmap_cmap.set_bad(heatmap_bg_color)
            ax_hm.set_facecolor(heatmap_bg_color)
            self.plots['ra_occ_hm'] = ax_hm.imshow(
                np.ma.masked_all((100, 100)), aspect='equal', origin='lower',
                cmap=heatmap_cmap, interpolation='bilinear')
            self.plots['ra_occ_cbar'] = self.figure.colorbar(
                self.plots['ra_occ_hm'], ax=ax_hm, fraction=0.046, pad=0.04)
            self.plots['ra_occ_cbar'].set_label('Power')

            # ---- 定位图峰值标记: 常驻 artist, 只切可见性, 不重建 figure ----
            # scalex/scaley=False: 新增 artist 不参与 autoscale, 否则会把坐标范围撑坏
            self._ra_occ_last_extent = None
            self._ra_occ_marker_last = None
            self._ra_occ_marker_ts = 0.0
            self.plots['ra_occ_view'] = bool(params.get('ra_occ_show_localization', False))
            self.plots['ra_occ_marker'], = ax_hm.plot(
                [], [], marker='o', linestyle='none',
                markersize=float(params.get('ra_occ_marker_size', 8.0)),
                markerfacecolor='red', markeredgecolor='red',
                zorder=5,                      # 压在座位椭圆/标号之上
                scalex=False, scaley=False)
            self.plots['ra_occ_marker'].set_visible(self.plots['ra_occ_view'])
            self.plots['ra_occ_oa_button'] = patches.Rectangle(
                (0.72, 1.21), 0.24, 0.065, transform=ax_hm.transAxes,
                facecolor='green', edgecolor='#222222', linewidth=1.2,
                clip_on=False, zorder=6)
            ax_hm.add_patch(self.plots['ra_occ_oa_button'])
            self.plots['ra_occ_oa_text'] = ax_hm.text(
                0.84, 1.243, "OUT", transform=ax_hm.transAxes,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', clip_on=False, zorder=7)
            self.plots['ra_occ_oa_filter_button'] = patches.Rectangle(
                (0.44, 1.21), 0.24, 0.065, transform=ax_hm.transAxes,
                facecolor='green', edgecolor='#222222', linewidth=1.2,
                clip_on=False, zorder=6)
            ax_hm.add_patch(self.plots['ra_occ_oa_filter_button'])
            self.plots['ra_occ_oa_filter_text'] = ax_hm.text(
                0.56, 1.243, "OUT", transform=ax_hm.transAxes,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', clip_on=False, zorder=7)

            ax_info.axis('off')
            self.plots['ra_occ_peak_info'] = ax_info.text(
                0.5, 0.5,
                "Mode[5] | Range: -- m | Angle: -- deg\n"
                "XY peak | X: -- m | Y: -- m\n"
                "Mean[5] | Range: -- m | Angle: -- deg",
                ha='center', va='center', fontsize=20, fontweight='bold',
                linespacing=1.25,
                transform=ax_info.transAxes)

            # 画座位椭圆 (Cartesian 坐标系, 复用 _draw_seating_ellipses)
            occ_params = params.copy()
            occ_params['occupancy_config'] = params.get('occupancy_config', None)
            self._draw_seating_ellipses(ax_hm, occ_params)

            # 按开关初始化显示模式 (定位图下椭圆/标号会被隐藏)
            self.set_ra_occ_view(self.plots['ra_occ_view'])

            # 右: 田字格状态面板 (2x2 grid)
            ax_occ.set_title("Seat Status")
            ax_occ.set_xlim(0, 1)
            ax_occ.set_ylim(0, 1)
            ax_occ.axis('off')

            # 田字格分割线
            ax_occ.plot([0.5, 0.5], [0, 1], 'k-', lw=2, transform=ax_occ.transAxes)
            ax_occ.plot([0, 1], [0.5, 0.5], 'k-', lw=2, transform=ax_occ.transAxes)

            # 4 座位置: TL=1(前排左), TR=2(前排右), BL=3(后排左), BR=4(后排右)
            cell_layout = {
                '1': (0.02, 0.52, 0.46, 0.44),   # 左上
                '2': (0.52, 0.52, 0.46, 0.44),   # 右上
                '3': (0.02, 0.02, 0.46, 0.44),   # 左下
                '4': (0.52, 0.02, 0.46, 0.44),   # 右下
            }

            self.plots['ra_occ_tian'] = {}
            for name, (cx, cy, cw, ch) in cell_layout.items():
                # 格子边框 (浅灰底色)
                border = patches.Rectangle(
                    (cx, cy), cw, ch, transform=ax_occ.transAxes,
                    facecolor='#f0f0f0', edgecolor='#888888',
                    linewidth=1.5, zorder=1)
                ax_occ.add_patch(border)

                # 座位名标签 (顶部居中)
                t_name = ax_occ.text(
                    cx + cw / 2, cy + ch - 0.06, f"Seat {name}",
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    transform=ax_occ.transAxes, zorder=3)

                # 状态色块 (中间区域)
                status_rect = patches.Rectangle(
                    (cx + 0.04, cy + 0.06), cw - 0.08, ch - 0.22,
                    transform=ax_occ.transAxes,
                    facecolor='green', edgecolor='none',
                    alpha=0.85, zorder=2)
                ax_occ.add_patch(status_rect)

                # 能量值/状态文字 (色块中央)
                t_energy = ax_occ.text(
                    cx + cw / 2, cy + 0.06 + (ch - 0.22) / 2,
                    "empty", ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white',
                    transform=ax_occ.transAxes, zorder=3)

                self.plots['ra_occ_tian'][name] = {
                    'border': border,
                    'name_text': t_name,
                    'status_rect': status_rect,
                    'energy_text': t_energy,
                }

            self.axes = {'main': ax_hm, 'info': ax_info, 'occ': ax_occ}
        elif mode == 'ANGLE-SPECTRUM':
            gs = self.figure.add_gridspec(1, 2, width_ratios=[3, 1])
            ax_as = self.figure.add_subplot(gs[0, 0])
            ax_rd = self.figure.add_subplot(gs[0, 1])

            # 左: 角度谱曲线图 (每个 CFAR 检出 bin 一条曲线)
            ax_as.set_title("DBF Angle Spectrum (CFAR detections)")
            ax_as.set_xlabel("Angle (°)")
            ax_as.set_ylabel("Power (dB)")
            ax_as.grid(True, linestyle=':', alpha=0.5)
            ax_as.set_xlim(-70, 70)
            # 曲线和峰值标记用空列表初始化, update_data 里动态更新
            self.plots['as_curves'] = []   # 角度谱曲线列表
            self.plots['as_peaks'] = []    # 峰值标记列表
            self.plots['as_labels'] = []   # 图例标签列表

            # 右: Range-Doppler 功率图 (看 CFAR 在哪些 bin 检出了目标)
            ax_rd.set_title("Range-Doppler (CFAR hits)")
            ax_rd.set_xlabel("Doppler bin")
            ax_rd.set_ylabel("Range bin")
            self.plots['rd_im'] = ax_rd.imshow(
                np.zeros((32, 64)), aspect='auto', origin='lower',
                cmap='jet', interpolation='bilinear'
            )

            self.axes = {'as': ax_as, 'rd': ax_rd}
        elif mode == 'AS-RAW':
            # 简单 1×2 布局: 左侧 DBF 角度谱曲线, 右侧 Range-Doppler 参考图
            gs = self.figure.add_gridspec(1, 2, width_ratios=[3, 1])
            ax_as = self.figure.add_subplot(gs[0, 0])
            ax_rd = self.figure.add_subplot(gs[0, 1])

            # 左: 每个 range bin 一条 DBF 角度谱曲线 (最强 doppler)
            ax_as.set_title("DBF Angle Spectrum (first N range bins, max Doppler)")
            ax_as.set_xlabel("Angle (°)")
            ax_as.set_ylabel("Power (dB)")
            ax_as.grid(True, linestyle=':', alpha=0.5)
            ax_as.set_xlim(-70, 70)
            self.plots['asr_curves'] = []
            self.plots['asr_peaks'] = []
            self.plots['asr_labels'] = []

            # 右: Range-Doppler 参考图
            ax_rd.set_title("Range-Doppler")
            ax_rd.set_xlabel("Doppler bin")
            ax_rd.set_ylabel("Range bin")
            self.plots['asr_rd'] = ax_rd.imshow(
                np.zeros((32, 64)), aspect='auto', origin='lower',
                cmap='jet', interpolation='bilinear')

            self.axes = {'as': ax_as, 'rd': ax_rd}
        self.canvas.draw()

    def set_ra_occ_view(self, localization):
        """切换 RA-OCCUPANCY 显示模式.

        localization=False → 热力图 (imshow + colorbar + 座位椭圆/标号)
        localization=True  → 定位图 (仅峰值处一个红色实心圆, 无椭圆无标号)

        只切 artist 可见性, 不重建 figure, 因此不会清空 ra_occ_peak_history
        与 App 侧 OA 模型窗口缓冲; 热力图下方 ax_info 的坐标文字始终保留.
        """
        localization = bool(localization)
        self.plots['ra_occ_view'] = localization

        hm = self.plots.get('ra_occ_hm')
        if hm is not None:
            hm.set_visible(not localization)

        cbar = self.plots.get('ra_occ_cbar')
        if cbar is not None:
            cbar.ax.set_visible(not localization)

        marker = self.plots.get('ra_occ_marker')
        if marker is not None:
            marker.set_visible(localization)
            if not localization:
                marker.set_data([], [])

        # 定位图下不保留座位椭圆与标号; 切回热力图时恢复
        for group in (getattr(self, '_seat_adult_patches', None),
                      getattr(self, '_seat_child_patches', None),
                      getattr(self, '_seat_texts', None)):
            for artist in (group or {}).values():
                artist.set_visible(not localization)

        # 锁定坐标范围, 保证隐藏 imshow 后视野与热力图一致
        ax = self.axes.get('main')
        ext = getattr(self, '_ra_occ_last_extent', None)
        if ax is not None and ext is not None:
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])

        self.canvas.draw_idle()

    def _update_ra_occ_marker(self, params, h_max, mean_th, peak_x_m, peak_y_m,
                              mode_range_m, mode_angle_deg,
                              mean_range_m, mean_angle_deg):
        """定位图红点位置更新 (含防抖: 时域平滑 + 保持).

        有效峰: 本轮 argmax 峰存在, 且原始 H 的最大值 h_max 不低于 oa_mean_threshold.
          有效时按 ra_occ_localization_smooth 取点, 与下方 ax_info 文字保持一致:
            'off'  → Cartesian 峰值      (对应文字 "XY peak" 行)
            'mode' → 最近 N 帧 polar 众数 (对应文字 "Mode[n]" 行)
            'mean' → 最近 N 帧 polar 均值 (对应文字 "Mean[n]" 行)
        无效峰: 若 hold_sec > 0, 红点在上一次有效位置保持 hold_sec 秒; 超时后清除.
        """
        marker = self.plots.get('ra_occ_marker')
        if marker is None:
            return

        show_loc = bool(params.get('ra_occ_show_localization', False))
        if show_loc != self.plots.get('ra_occ_view', False):
            self.set_ra_occ_view(show_loc)      # 参数被其它入口改动时也能跟上
        if not show_loc:
            return

        smooth = str(params.get('ra_occ_localization_smooth', 'mode')).lower()
        try:
            hold_sec = float(params.get('ra_occ_localization_hold_sec', 0.0) or 0.0)
        except (TypeError, ValueError):
            hold_sec = 0.0

        peak_valid = (peak_x_m is not None and peak_y_m is not None
                      and float(h_max) >= float(mean_th))

        xy = None
        if peak_valid:
            xy = self._ra_occ_smoothed_xy(smooth, peak_x_m, peak_y_m,
                                          mode_range_m, mode_angle_deg,
                                          mean_range_m, mean_angle_deg)
            self._ra_occ_marker_last = xy
            self._ra_occ_marker_ts = time.time()
        elif (hold_sec > 0 and self._ra_occ_marker_last is not None
              and (time.time() - self._ra_occ_marker_ts) <= hold_sec):
            xy = self._ra_occ_marker_last      # 保持上一次有效位置
        else:
            self._ra_occ_marker_last = None

        if xy is None:
            marker.set_data([], [])
        else:
            marker.set_data([xy[0]], [xy[1]])

    @staticmethod
    def _ra_occ_smoothed_xy(smooth, peak_x_m, peak_y_m,
                            mode_range_m, mode_angle_deg,
                            mean_range_m, mean_angle_deg):
        """按平滑策略给出红点坐标; polar 输入按 x=r*sin(θ), y=-r*cos(θ) 换算.

        换算式与 _ra_to_cartesian / _seat_ra_energy 中的映射完全一致.
        """
        def polar_to_xy(r_m, a_deg):
            a = np.deg2rad(a_deg)
            return (float(r_m * np.sin(a)), float(-r_m * np.cos(a)))

        if smooth == 'mode' and mode_range_m is not None and mode_angle_deg is not None:
            return polar_to_xy(mode_range_m, mode_angle_deg)
        if smooth == 'mean' and mean_range_m is not None and mean_angle_deg is not None:
            return polar_to_xy(mean_range_m, mean_angle_deg)
        return (float(peak_x_m), float(peak_y_m))

    def _update_ra_occ_oa_filter(self, oa_label, oa_status, params):
        """Apply the configurable majority filter to raw OA button values."""
        try:
            window_size = max(1, int(params.get('oa_filter_window_size', 5)))
        except (TypeError, ValueError):
            window_size = 5
        try:
            in_threshold = int(params.get('oa_filter_in_threshold', 3))
        except (TypeError, ValueError):
            in_threshold = 3
        in_threshold = max(1, min(window_size, in_threshold))

        if self.ra_occ_oa_filter_history.maxlen != window_size:
            self.ra_occ_oa_filter_history = deque(
                list(self.ra_occ_oa_filter_history)[-window_size:],
                maxlen=window_size)

        raw_status = str(oa_status or 'out').lower()
        try:
            raw_is_in = int(oa_label or 0) == 1
        except (TypeError, ValueError):
            raw_is_in = False
        self.ra_occ_oa_filter_history.append(1 if raw_is_in else 0)
        history = self.ra_occ_oa_filter_history
        in_count = sum(history)

        # Wait for one complete window before allowing a filtered IN result.
        if len(history) >= window_size and in_count >= in_threshold:
            filtered_label, filtered_status = 1, 'in'
        else:
            filtered_label = 0
            filtered_status = 'empty' if raw_status == 'empty' else 'out'

        if 'ra_occ_oa_filter_button' in self.plots:
            color = 'red' if filtered_label == 1 else 'green'
            text = 'IN' if filtered_label == 1 else (
                'EMPTY' if filtered_status == 'empty' else 'OUT')
            self.plots['ra_occ_oa_filter_button'].set_facecolor(color)
            self.plots['ra_occ_oa_filter_text'].set_text(text)
        return filtered_label, filtered_status

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
        elif mode in ('POINT-CLOUD', 'POINT-CLOUD-OPTIMIZED', 'POINT-CLOUD-DUBHE', 'POINT-CLOUD-PAPER', 'RA-CFAR'):
            if not data or 'detected_points' not in data: return
            p = data['params']
            breath_val = data.get('breath_val', 0.0)
            mode_titles = {
                'POINT-CLOUD': "Vehicle Occupancy Point Cloud (CA-CFAR)",
                'POINT-CLOUD-OPTIMIZED': "Vehicle Occupancy Point Cloud (Optimized)",
                'POINT-CLOUD-DUBHE': "Vehicle Occupancy Point Cloud (Dubhe CPD)",
                'POINT-CLOUD-PAPER': "Vehicle Occupancy Point Cloud (Multipass CFAR, IEEE JSEN'24)",
                'RA-CFAR': "Vehicle Occupancy (Range-Azimuth Two-Pass CFAR)",
            }
            base_title = mode_titles.get(mode, "Point Cloud")
            # RA-CFAR 不需要历史点轨迹, 直接替换
            if mode == 'RA-CFAR':
                self.pc_history.clear()
            info_text = f"Breath: {breath_val:.3f}"
            if 'filename' in data:
                self.axes['main'].set_title(f"{base_title} ({info_text})\nFile: {data['filename']}", fontsize=10)
            else:
                self.axes['main'].set_title(f"{base_title}\n{info_text}")

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

            # 座位占用状态着色
            occupancy = data.get('occupancy', None)
            if occupancy is not None:
                self._update_seat_colors(occupancy)
                occ_str = '|'.join(f"{n}={occupancy.get(n, 0)}" for n in ['1','2','3','4'])
                f_vals = data.get('f_values', {})
                f_str = '|'.join(f"{n}={f_vals.get(n, 0):.3f}" for n in ['1','2','3','4'])
                occ_title = f"Occ: [{occ_str}]  f_k: [{f_str}]"
                current_title = self.axes['main'].get_title()
                if 'Occ:' not in current_title:
                    self.axes['main'].set_title(f"{current_title}\n{occ_title}")

            self.canvas.draw_idle() # 使用 draw_idle 提高响应速度
        elif mode == 'RA-HEATMAP':
            H_db = data['heatmap']
            ranges_m = data['ranges_m']
            angles_deg = data['angles_deg']
            self.plots['ra_hm'].set_data(H_db)
            self.plots['ra_hm'].set_extent([angles_deg[0], angles_deg[-1],
                                            ranges_m[0], ranges_m[-1]])
            vmin = np.min(H_db)
            vmax = np.max(H_db)
            self.plots['ra_hm'].set_clim(vmin=vmin, vmax=vmax)
            self.axes['main'].set_title(
                f"Capon Range-Azimuth Heatmap | vmax={vmax:.1f}dB")
            self.canvas.draw_idle()
        elif mode == 'RA-OCCUPANCY':
            H_cart = data['heatmap']          # Cartesian remapped
            xs_cart = data['xs_cart']
            ys_cart = data['ys_cart']
            energy = data.get('energy', {})
            occupancy = data.get('occupancy', {})

            # === 左: Cartesian RA 热力图 ===
            self.plots['ra_occ_hm'].set_data(np.ma.masked_invalid(H_cart))
            self.plots['ra_occ_hm'].set_extent([xs_cart[0], xs_cart[-1],
                                                ys_cart[0], ys_cart[-1]])
            p = data.get('params', {})
            clim_mode = p.get('heatmap_clim_mode', 'auto')
            if clim_mode == 'fixed':
                vmin = p.get('heatmap_clim_vmin', -80)
                vmax = p.get('heatmap_clim_vmax', 0)
            else:
                valid_values = H_cart[np.isfinite(H_cart)]
                if valid_values.size:
                    vmin = max(-80, np.percentile(valid_values, 5))
                    vmax = np.max(valid_values)
                else:
                    vmin, vmax = 0.0, 1.0
            if vmin >= vmax:
                vmin = vmax - 1.0
            self.plots['ra_occ_hm'].set_clim(vmin=vmin, vmax=vmax)

            peak_range_m = data.get('peak_range_m')
            peak_angle_deg = data.get('peak_angle_deg')
            peak_x_m = data.get('peak_x_m')
            peak_y_m = data.get('peak_y_m')
            mode_range_m = mode_angle_deg = None
            mean_range_m = mean_angle_deg = None
            if all(value is not None for value in
                   (peak_range_m, peak_angle_deg, peak_x_m, peak_y_m)):
                self.ra_occ_peak_history.append(
                    (float(peak_range_m), float(peak_angle_deg)))
                recent_peaks = list(self.ra_occ_peak_history)

                def latest_mode(values):
                    counts = {value: values.count(value) for value in values}
                    max_count = max(counts.values())
                    return next(value for value in reversed(values)
                                if counts[value] == max_count)

                recent_ranges = [peak[0] for peak in recent_peaks]
                recent_angles = [peak[1] for peak in recent_peaks]
                mode_range_m = latest_mode(recent_ranges)
                mode_angle_deg = latest_mode(recent_angles)
                mean_range_m = float(np.mean(recent_ranges))
                mean_angle_deg = float(np.mean(recent_angles))
                sample_count = len(recent_peaks)
                peak_text = (
                    f"Mode[{sample_count}] | Range: {mode_range_m:.2f} m | "
                    f"Angle: {mode_angle_deg:+.1f} deg\n"
                    f"XY peak | X: {peak_x_m:+.2f} m | "
                    f"Y: {peak_y_m:+.2f} m\n"
                    f"Mean[{sample_count}] | Range: {mean_range_m:.2f} m | "
                    f"Angle: {mean_angle_deg:+.1f} deg")
            else:
                peak_text = (
                    "Mode[5] | Range: -- m | Angle: -- deg\n"
                    "XY peak | X: -- m | Y: -- m\n"
                    "Mean[5] | Range: -- m | Angle: -- deg")
            self.plots['ra_occ_peak_info'].set_text(peak_text)

            h_max = float(data.get('h_max', 0.0))
            mean_th = float(data.get('oa_mean_threshold', p.get('oa_mean_threshold', 0.0)))
            oa_label = data.get('oa_label', 0)
            oa_score = data.get('oa_score', None)
            oa_status = data.get('oa_status', 'out')
            score_text = "" if oa_score is None else f" | score={oa_score:.3f}"

            # The filtered OA indicator is computed independently from the raw indicator below.
            self._update_ra_occ_oa_filter(oa_label, oa_status, p)
            if 'ra_occ_oa_button' in self.plots:
                color = 'red' if oa_label == 1 else 'green'
                text = 'IN' if oa_label == 1 else ('EMPTY' if oa_status == 'empty' else 'OUT')
                self.plots['ra_occ_oa_button'].set_facecolor(color)
                self.plots['ra_occ_oa_text'].set_text(text)

            # ---- 定位图: 仅在峰值处画一个红色实心圆 (防抖: 时域平滑 + 保持) ----
            self._update_ra_occ_marker(p, h_max, mean_th, peak_x_m, peak_y_m,
                                       mode_range_m, mode_angle_deg,
                                       mean_range_m, mean_angle_deg)

            # 座位椭圆着色 (复用 _update_seat_colors)
            if occupancy:
                self._update_seat_colors(occupancy)

            # 标题: 能量和 + 占位状态
            occ_parts = []
            energy_parts = []
            max_main_e = 0.0
            for seat_name in sorted(energy.keys(), key=lambda n: int(n)) if energy else []:
                occ_parts.append(f"{seat_name}={occupancy.get(seat_name, 0)}")
                e = energy.get(seat_name, {})
                if isinstance(e, dict):
                    m_e = e.get('main', 0.0)
                    s_e = e.get('child_special', 0.0)
                    energy_parts.append(f"{seat_name} M={m_e:.4f} S={s_e:.4f}")
                    max_main_e = max(max_main_e, m_e)
                else:
                    energy_parts.append(f"{seat_name}={e:.4f}")
                    max_main_e = max(max_main_e, e)
            self.axes['main'].set_title(
                f"RA Occupancy (Cartesian)\n"
                f"max mainE={max_main_e:.4f} | "
                f"Occ: [{'|'.join(occ_parts)}]\n"
                f"E: [{'|'.join(energy_parts)}]\n"
                f"H max={h_max:.4f} | th={mean_th:.4f}{score_text}",
                fontsize=9,
                loc='left')

            # === 右: 田字格状态面板 (原地更新色块+文字, 不复绘) ===
            state = data.get('state', {})  # 0=empty, 1=child, 2=adult
            state_colors = {0: 'green', 1: 'gold', 2: 'red'}
            state_labels = {0: 'empty', 1: 'child', 2: 'ADULT'}

            for name, cell in self.plots.get('ra_occ_tian', {}).items():
                s_val = state.get(name, 0)
                e = energy.get(name, {})
                if isinstance(e, dict):
                    m_e = e.get('main', 0.0)
                    s_e = e.get('child_special', 0.0)
                    e_display = f"M={m_e:.4f}\nS={s_e:.4f}"
                else:
                    e_display = f"E={e:.4f}"
                color = state_colors.get(s_val, 'green')
                label = state_labels.get(s_val, 'empty')

                cell['status_rect'].set_facecolor(color)
                cell['energy_text'].set_text(f"{label}\n{e_display}")
                # 金色背景用深色文字
                text_color = 'black' if s_val == 1 else 'white'
                cell['energy_text'].set_color(text_color)

            self.canvas.draw_idle()
        elif mode == 'ANGLE-SPECTRUM':
            spectra = data['spectra']
            power_map = data['power_map']

            # 左: 角度谱曲线 —— 清除旧曲线, 重新画
            ax = self.axes['as']
            for line in self.plots['as_curves']:
                line.remove()
            for pk in self.plots['as_peaks']:
                pk.remove()
            for txt in self.plots['as_labels']:
                txt.remove()
            self.plots['as_curves'] = []
            self.plots['as_peaks'] = []
            self.plots['as_labels'] = []

            colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(spectra))))
            for i, sp in enumerate(spectra):
                line, = ax.plot(sp['angles_deg'], sp['pwr_db'],
                                color=colors[i], alpha=0.8, linewidth=1.5)
                self.plots['as_curves'].append(line)

                # 标记最强峰位置
                peak_idx = np.argmax(sp['pwr_linear'])
                peak_angle = sp['angles_deg'][peak_idx]
                peak_pwr_db = sp['pwr_db'][peak_idx]
                pk, = ax.plot(peak_angle, peak_pwr_db, 'x',
                              color=colors[i], markersize=10, mew=2)
                self.plots['as_peaks'].append(pk)

                # 标签: 距离 + SNR
                label = ax.annotate(
                    f"R={sp['range_m']:.2f}m\nθ={peak_angle:.1f}°\n{sp['snr_db']:.1f}dB",
                    xy=(peak_angle, peak_pwr_db),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=7, color=colors[i],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                self.plots['as_labels'].append(label)

            n_det = len(spectra)
            ax.set_title(f"DBF Angle Spectrum ({n_det} CFAR detection{'s' if n_det>1 else ''})")
            # 动态 y 轴范围
            if spectra:
                all_db = np.concatenate([s['pwr_db'] for s in spectra])
                y_min = max(-40, np.percentile(all_db, 5) - 5)
                y_max = np.max(all_db) + 3
                ax.set_ylim(y_min, y_max)

            # 右: Range-Doppler 参考图
            self.plots['rd_im'].set_data(power_map)
            if np.max(power_map) > 0:
                self.plots['rd_im'].set_clim(vmin=0, vmax=np.max(power_map))

            self.canvas.draw_idle()
        elif mode == 'AS-RAW':
            spectra = data['spectra']         # list of dicts: 每条 range bin 一个
            power_map = data['power_map']     # [n_range, n_dop] dB

            # === 左: DBF 角度谱曲线 (每个 range bin 一条) ===
            ax = self.axes['as']
            for line in self.plots['asr_curves']:
                line.remove()
            for pk in self.plots['asr_peaks']:
                pk.remove()
            for txt in self.plots['asr_labels']:
                txt.remove()
            self.plots['asr_curves'] = []
            self.plots['asr_peaks'] = []
            self.plots['asr_labels'] = []

            colors = plt.cm.viridis(np.linspace(0.15, 0.95, max(1, len(spectra))))
            for i, sp in enumerate(spectra):
                line, = ax.plot(sp['angles_deg'], sp['pwr_db'],
                                color=colors[i], alpha=0.85, linewidth=1.5)
                self.plots['asr_curves'].append(line)

                # 标记最强峰
                peak_idx = np.argmax(sp['pwr_linear'])
                peak_angle = sp['angles_deg'][peak_idx]
                peak_pwr_db = sp['pwr_db'][peak_idx]
                pk, = ax.plot(peak_angle, peak_pwr_db, 'x',
                              color=colors[i], markersize=10, mew=2)
                self.plots['asr_peaks'].append(pk)

                # 标签: range_bin + distance + peak angle
                txt = ax.annotate(
                    f"R{sp['range_bin']} {sp['range_m']:.2f}m\nθ={peak_angle:.1f}° D{sp['doppler_bin']}",
                    xy=(peak_angle, peak_pwr_db),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=7, color=colors[i],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                self.plots['asr_labels'].append(txt)

            n_bins = len(spectra)
            ax.set_title(f"DBF Angle Spectrum ({n_bins} range bins, max Doppler, no CFAR)")
            # 动态 y 轴
            if spectra:
                all_db = np.concatenate([s['pwr_db'] for s in spectra])
                y_min = max(-40, np.percentile(all_db, 5) - 5)
                y_max = np.max(all_db) + 3
                ax.set_ylim(y_min, y_max)

            # === 右: Range-Doppler 参考图 ===
            self.plots['asr_rd'].set_data(power_map)
            vmin_rd = max(-20, np.percentile(power_map, 5))
            vmax_rd = np.max(power_map)
            self.plots['asr_rd'].set_clim(vmin=vmin_rd, vmax=vmax_rd)

            self.canvas.draw_idle()
        self.canvas.draw()

    def _draw_seating_ellipses(self, ax, params):
        """
        根据 SEAT-OCCUPANCY 配置绘制座椅椭圆, 支持动态着色.
        主检测区: 实线, 粗线, 标号;  特判区: 虚线, 细线, 标号+s.
        若配置不存在则回退到论文 Table II 硬编码值.
        """
        occ_cfg = params.get('occupancy_config', None)
        self._has_child_ellipses = False
        self._seat_adult_patches = {}
        self._seat_child_patches = {}
        self._seat_texts = {}

        if occ_cfg:
            seat_type = occ_cfg.get('seat_type', '4_seats')
            key = 'seats_4' if seat_type == '4_seats' else 'seats_5'
            seat_defs = occ_cfg.get(key, occ_cfg.get('seats_4', []))

            for sd in seat_defs:
                name = sd['name']
                # --- 主检测区椭圆 (main): 实线 ---
                main = sd.get('main', sd)
                a_ellipse = patches.Ellipse(
                    (main['cx'], main['cy']),
                    width=main['rx'] * 2, height=main['ry'] * 2,
                    edgecolor='green', facecolor='none',
                    linestyle='-', linewidth=2.5, alpha=0.8
                )
                ax.add_patch(a_ellipse)
                self._seat_adult_patches[name] = a_ellipse

                # --- 特判区椭圆 (child_special): 虚线 (仅在有 child_special 子配置时) ---
                child_special = sd.get('child_special', None)
                if child_special is not None:
                    self._has_child_ellipses = True
                    c_ellipse = patches.Ellipse(
                        (child_special['cx'], child_special['cy']),
                        width=child_special['rx'] * 2, height=child_special['ry'] * 2,
                        edgecolor='green', facecolor='none',
                        linestyle='--', linewidth=1.5, alpha=0.6
                    )
                    ax.add_patch(c_ellipse)
                    self._seat_child_patches[name] = c_ellipse

                # 主检测区椭圆中心标号
                txt = ax.text(main['cx'], main['cy'], name,
                              color='green', ha='center', fontweight='bold', fontsize=9)
                self._seat_texts[name] = txt
        else:
            # fallback: 原硬编码逻辑 (无小孩椭圆)
            is_5 = (params.get('seat_type') == '5_seats')
            if not is_5:
                seats = [
                    {'center': (-0.3, -0.6), 'rx': 0.2, 'ry': 0.2, 'name': '1'},
                    {'center': (0.3, -0.6),  'rx': 0.2, 'ry': 0.2, 'name': '2'},
                    {'center': (-0.3, -1.4), 'rx': 0.2, 'ry': 0.2, 'name': '3'},
                    {'center': (0.3, -1.4),  'rx': 0.2, 'ry': 0.2, 'name': '4'},
                ]
            else:
                seats = [
                    {'center': (-0.3, -0.5), 'rx': 0.25, 'ry': 0.2, 'name': '1'},
                    {'center': (0.3, -0.5),  'rx': 0.25, 'ry': 0.2, 'name': '2'},
                    {'center': (-0.4, -1.3), 'rx': 0.25, 'ry': 0.2, 'name': '3'},
                    {'center': (0, -1.3),    'rx': 0.25, 'ry': 0.2, 'name': '5'},
                    {'center': (0.4, -1.3),  'rx': 0.25, 'ry': 0.2, 'name': '4'},
                ]
            for s in seats:
                ellipse = patches.Ellipse(
                    s['center'], width=s['rx'] * 2, height=s['ry'] * 2,
                    edgecolor='green', facecolor='none',
                    linestyle='-', linewidth=2.0, alpha=0.8
                )
                ax.add_patch(ellipse)
                txt = ax.text(s['center'][0], s['center'][1], s['name'],
                              color='green', ha='center', fontweight='bold')
                self._seat_adult_patches[s['name']] = ellipse
                self._seat_texts[s['name']] = txt

    def _update_seat_colors(self, occupancy):
        """根据三态占用状态更新椭圆颜色: 2=成人=红, 1=儿童=金, 0=空闲=绿.
        成人椭圆始终为实线, 小孩椭圆始终为虚线."""
        # 兼容旧的 _seat_patches 属性 (其他模式如 POINT-CLOUD)
        if hasattr(self, '_seat_patches') and self._seat_patches:
            for name, patch in self._seat_patches.items():
                occ = occupancy.get(name, 0) if isinstance(occupancy, dict) else 0
                if occ == 2:
                    color, lw = 'red', 3.0
                elif occ == 1:
                    color, lw = 'gold', 2.5
                else:
                    color, lw = 'green', 2.0
                patch.set_edgecolor(color)
                patch.set_linewidth(lw)
                if name in self._seat_texts:
                    self._seat_texts[name].set_color(color)
            return

        # 新的双椭圆模式
        if not hasattr(self, '_seat_adult_patches') or not self._seat_adult_patches:
            return
        for name, a_patch in self._seat_adult_patches.items():
            occ = occupancy.get(name, 0) if isinstance(occupancy, dict) else 0
            c_patch = self._seat_child_patches.get(name) if hasattr(self, '_seat_child_patches') else None

            if occ == 2:          # 成人: 成人圈红粗, 小孩圈灰细
                a_patch.set_edgecolor('red')
                a_patch.set_linewidth(3.0)
                if c_patch:
                    c_patch.set_edgecolor('#cccccc')
                    c_patch.set_linewidth(1.0)
            elif occ == 1:        # 儿童: 成人圈绿, 小孩圈金
                a_patch.set_edgecolor('green')
                a_patch.set_linewidth(2.0)
                if c_patch:
                    c_patch.set_edgecolor('gold')
                    c_patch.set_linewidth(2.0)
            else:                 # 空闲: 两圈皆绿
                a_patch.set_edgecolor('green')
                a_patch.set_linewidth(2.0)
                if c_patch:
                    c_patch.set_edgecolor('green')
                    c_patch.set_linewidth(1.5)

            if name in self._seat_texts:
                self._seat_texts[name].set_color(
                    'red' if occ == 2 else ('gold' if occ == 1 else 'green'))

# ==============================================================================
# 6.6 座椅配置对话框
# ==============================================================================
class SeatConfigDialog(tk.Toplevel):
    """座椅占用检测参数配置面板，每座椅独立调节"""

    def __init__(self, parent, occ_params, callback):
        super().__init__(parent)
        self.title("座椅占用检测配置")
        self.minsize(950, 480)
        self.resizable(True, True)
        self.occ_params = occ_params.copy()
        self.callback = callback

        self._build_ui()
        self._load_params()

    def _build_ui(self):
        # 顶层: 使能 + 车型 + 平滑参数
        top = tk.LabelFrame(self, text="全局设置", padx=10, pady=5)
        top.pack(fill='x', padx=10, pady=5)

        self.var_enable = tk.BooleanVar()
        tk.Checkbutton(top, text="启用座椅占用检测", variable=self.var_enable).grid(row=0, column=0, sticky='w')

        tk.Label(top, text="车型:").grid(row=0, column=1, padx=(20, 5))
        self.var_seat_type = tk.StringVar(value='4_seats')
        ttk.Combobox(top, textvariable=self.var_seat_type,
                     values=('4_seats', '5_seats'), width=8, state='readonly').grid(row=0, column=2)

        tk.Label(top, text="平滑帧数:").grid(row=0, column=3, padx=(20, 5))
        self.var_smooth = tk.StringVar(value='3')
        tk.Spinbox(top, textvariable=self.var_smooth, from_=1, to=10, width=4).grid(row=0, column=4)

        tk.Label(top, text="平滑阈值:").grid(row=0, column=5, padx=(10, 5))
        self.var_smooth_th = tk.StringVar(value='0.5')
        tk.Spinbox(top, textvariable=self.var_smooth_th, from_=0.1, to=1.0, increment=0.1, width=4).grid(row=0, column=6)

        tk.Label(top, text="保持时长(s):").grid(row=0, column=7, padx=(10, 5))
        self.var_hold = tk.StringVar(value='2.0')
        tk.Spinbox(top, textvariable=self.var_hold, from_=0.0, to=30.0, increment=0.5, width=5).grid(row=0, column=8)

        # 座椅参数卡片 (4 或 5 个)
        seat_frame = tk.LabelFrame(self, text="座椅参数 (M=主检测区 S=特判区: cx/cy=坐标, rx/ry=半轴, A/C-阈=成人/娃娃阈值)", padx=10, pady=5)
        seat_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # 画布+滚动条 (水平+垂直)
        canvas = tk.Canvas(seat_frame, height=280)
        h_scrollbar = ttk.Scrollbar(seat_frame, orient='horizontal', command=canvas.xview)
        v_scrollbar = ttk.Scrollbar(seat_frame, orient='vertical', command=canvas.yview)
        self.seat_inner = tk.Frame(canvas)
        self.seat_inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.seat_inner, anchor='nw')
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        seat_frame.rowconfigure(0, weight=1)
        seat_frame.columnconfigure(0, weight=1)

        # 表头: 14列 (M=主检测区, A=成人阈值, C=娃娃阈值, S=特判区角坑)
        headers = ['座椅', '名称', 'ratio',
                   'M-cx', 'M-cy', 'M-rx', 'M-ry', 'A-阈',
                   'C-阈下', 'C-阈上',
                   'S-ccx', 'S-ccy', 'S-crx', 'S-cry']
        col_widths = [5, 4, 5] + [4]*11
        for j, (h, w) in enumerate(zip(headers, col_widths)):
            tk.Label(self.seat_inner, text=h, font=('Arial', 7, 'bold'),
                     width=w, anchor='center', relief='ridge', bg='#e0e0e0').grid(row=0, column=j, padx=1, pady=1)

        # 为 5 个座椅各建一行控件 (4座显示前4行, 5座显示全部)
        self.seat_vars = []
        for i in range(5):
            tk.Label(self.seat_inner, text=f"座椅{i+1}", anchor='center', font=('Arial', 8)).grid(row=i+1, column=0, padx=1, pady=1)

            v_name = tk.StringVar(value=str(i+1))
            tk.Entry(self.seat_inner, textvariable=v_name, width=4).grid(row=i+1, column=1, padx=1)

            v_ra_ratio = tk.StringVar(value='0.3')
            tk.Spinbox(self.seat_inner, textvariable=v_ra_ratio, from_=0.0, to=1.0, increment=0.05, width=5).grid(row=i+1, column=2, padx=1)

            # Main 主检测区参数 (cx, cy, rx, ry)
            v_m_cx = tk.StringVar(value='0.0')
            tk.Spinbox(self.seat_inner, textvariable=v_m_cx, from_=-3.0, to=3.0, increment=0.05, width=5).grid(row=i+1, column=3, padx=1)

            v_m_cy = tk.StringVar(value='0.0')
            tk.Spinbox(self.seat_inner, textvariable=v_m_cy, from_=-3.0, to=0.0, increment=0.05, width=5).grid(row=i+1, column=4, padx=1)

            v_m_rx = tk.StringVar(value='0.2')
            tk.Spinbox(self.seat_inner, textvariable=v_m_rx, from_=0.05, to=1.0, increment=0.01, width=5).grid(row=i+1, column=5, padx=1)

            v_m_ry = tk.StringVar(value='0.2')
            tk.Spinbox(self.seat_inner, textvariable=v_m_ry, from_=0.05, to=1.0, increment=0.01, width=5).grid(row=i+1, column=6, padx=1)

            # Adult 阈值
            v_a_th = tk.StringVar(value='0.02')
            tk.Spinbox(self.seat_inner, textvariable=v_a_th, from_=0.0, to=1.0, increment=0.001, width=5).grid(row=i+1, column=7, padx=1)

            # Child 阈值 (基于主检测区)
            v_c_th_lo = tk.StringVar(value='0.05')
            tk.Spinbox(self.seat_inner, textvariable=v_c_th_lo, from_=0.0, to=100.0, increment=0.01, width=5).grid(row=i+1, column=8, padx=1)

            v_c_th_hi = tk.StringVar(value='5.0')
            tk.Spinbox(self.seat_inner, textvariable=v_c_th_hi, from_=0.0, to=100.0, increment=0.1, width=5).grid(row=i+1, column=9, padx=1)

            # Child Special 特判区参数 (角坑兜底: cx, cy, rx, ry)
            v_s_cx = tk.StringVar(value='0.0')
            tk.Spinbox(self.seat_inner, textvariable=v_s_cx, from_=-3.0, to=3.0, increment=0.05, width=5).grid(row=i+1, column=10, padx=1)

            v_s_cy = tk.StringVar(value='0.0')
            tk.Spinbox(self.seat_inner, textvariable=v_s_cy, from_=-3.0, to=0.0, increment=0.05, width=5).grid(row=i+1, column=11, padx=1)

            v_s_rx = tk.StringVar(value='0.15')
            tk.Spinbox(self.seat_inner, textvariable=v_s_rx, from_=0.05, to=1.0, increment=0.01, width=5).grid(row=i+1, column=12, padx=1)

            v_s_ry = tk.StringVar(value='0.15')
            tk.Spinbox(self.seat_inner, textvariable=v_s_ry, from_=0.05, to=1.0, increment=0.01, width=5).grid(row=i+1, column=13, padx=1)

            # Child Special 阈值 (特判区专用)
            v_s_th_lo = tk.StringVar(value='0.05')
            tk.Spinbox(self.seat_inner, textvariable=v_s_th_lo, from_=0.0, to=100.0, increment=0.01, width=5).grid(row=i+1, column=14, padx=1)

            v_s_th_hi = tk.StringVar(value='5.0')
            tk.Spinbox(self.seat_inner, textvariable=v_s_th_hi, from_=0.0, to=100.0, increment=0.1, width=5).grid(row=i+1, column=15, padx=1)

            self.seat_vars.append({
                'name': v_name, 'ra_peak_ratio': v_ra_ratio,
                'main': {'cx': v_m_cx, 'cy': v_m_cy, 'rx': v_m_rx, 'ry': v_m_ry},
                'adult_threshold': v_a_th,
                'child_threshold_low': v_c_th_lo, 'child_threshold_high': v_c_th_hi,
                'child_special': {'cx': v_s_cx, 'cy': v_s_cy, 'rx': v_s_rx, 'ry': v_s_ry,
                                  'threshold_low': v_s_th_lo, 'threshold_high': v_s_th_hi},
            })

        # 底部按钮
        btn_row = tk.Frame(self, pady=10)
        btn_row.pack(fill='x', padx=10)
        tk.Button(btn_row, text="  保存  ", bg='#cfc', width=10,
                  command=self._save).pack(side='left', padx=(20, 10))
        tk.Button(btn_row, text=" 取消 ", width=10,
                  command=self.destroy).pack(side='left', padx=10)
        tk.Label(btn_row, text="修改后点击保存即生效，椭圆会实时刷新",
                 fg='gray').pack(side='right', padx=10)

    def _load_params(self):
        """从 occ_params 字典加载到 UI 控件"""
        p = self.occ_params
        self.var_enable.set(p.get('enable', True))
        self.var_seat_type.set(p.get('seat_type', '4_seats'))
        self.var_smooth.set(str(p.get('smooth_window', 3)))
        self.var_smooth_th.set(str(p.get('smooth_threshold', 0.5)))
        self.var_hold.set(str(p.get('hold_time_sec', 2.0)))

        key = 'seats_4' if self.var_seat_type.get() == '4_seats' else 'seats_5'
        seats = p.get(key, p.get('seats_4', []))
        for i, sv in enumerate(self.seat_vars):
            if i < len(seats):
                s = seats[i]
                sv['name'].set(s.get('name', str(i+1)))
                sv['ra_peak_ratio'].set(str(s.get('ra_peak_ratio', 0.3)))
                # main 区域
                main = s.get('main', s)
                sv['main']['cx'].set(str(main.get('cx', 0.0)))
                sv['main']['cy'].set(str(main.get('cy', 0.0)))
                sv['main']['rx'].set(str(main.get('rx', 0.2)))
                sv['main']['ry'].set(str(main.get('ry', 0.2)))
                # 阈值
                sv['adult_threshold'].set(str(s.get('adult_threshold', 0.02)))
                sv['child_threshold_low'].set(str(s.get('child_threshold_low', 0.05)))
                sv['child_threshold_high'].set(str(s.get('child_threshold_high', 5.0)))
                # child_special 特判区
                spec = s.get('child_special', s)
                sv['child_special']['cx'].set(str(spec.get('cx', 0.0)))
                sv['child_special']['cy'].set(str(spec.get('cy', 0.0)))
                sv['child_special']['rx'].set(str(spec.get('rx', 0.15)))
                sv['child_special']['ry'].set(str(spec.get('ry', 0.15)))
                sv['child_special']['threshold_low'].set(str(spec.get('threshold_low', 0.05)))
                sv['child_special']['threshold_high'].set(str(spec.get('threshold_high', 5.0)))

    def _save(self):
        """从 UI 控件写回 occ_params 字典, 执行回调"""
        p = self.occ_params
        p['enable'] = self.var_enable.get()
        p['seat_type'] = self.var_seat_type.get()
        try:
            p['smooth_window'] = int(self.var_smooth.get())
            p['smooth_threshold'] = float(self.var_smooth_th.get())
            p['hold_time_sec'] = float(self.var_hold.get())
        except ValueError:
            pass

        key = 'seats_4' if p['seat_type'] == '4_seats' else 'seats_5'
        num_seats = 4 if p['seat_type'] == '4_seats' else 5
        seat_list = []
        for i in range(num_seats):
            sv = self.seat_vars[i]
            try:
                seat_list.append({
                    'name': sv['name'].get(),
                    'ra_peak_ratio': float(sv['ra_peak_ratio'].get()),
                    'main': {
                        'cx': float(sv['main']['cx'].get()),
                        'cy': float(sv['main']['cy'].get()),
                        'rx': float(sv['main']['rx'].get()),
                        'ry': float(sv['main']['ry'].get()),
                    },
                    'adult_threshold': float(sv['adult_threshold'].get()),
                    'child_threshold_low': float(sv['child_threshold_low'].get()),
                    'child_threshold_high': float(sv['child_threshold_high'].get()),
                    'child_special': {
                        'cx': float(sv['child_special']['cx'].get()),
                        'cy': float(sv['child_special']['cy'].get()),
                        'rx': float(sv['child_special']['rx'].get()),
                        'ry': float(sv['child_special']['ry'].get()),
                        'threshold_low': float(sv['child_special']['threshold_low'].get()),
                        'threshold_high': float(sv['child_special']['threshold_high'].get()),
                    },
                })
            except ValueError:
                continue
        p[key] = seat_list

        self.callback(p)
        self.destroy()


def fit_to_screen(root, pref_w=1300, pref_h=850,
                  margin_w=60, margin_h=80, min_w=880, min_h=560):
    """按屏幕可用尺寸自适应窗口大小并居中, 避免小屏/高 DPI 下显示不全.

    窗口不会超过屏幕; 但若内容仍高于可用高度, 由左侧 ScrollFrame 滚动承载.
    """
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w = max(min_w, min(pref_w, sw - margin_w))
    h = max(min_h, min(pref_h, sh - margin_h))
    root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
    root.minsize(min_w, min_h)
    return w, h


class ScrollFrame(tk.Frame):
    """通用可滚动容器: 内容控件的 parent 传 self.body 即可.

    用途: 左侧控制面板内容纵向需求远超小屏可用高度, tkinter 本身不会滚动,
    底部 START/STOP 等按钮会被裁掉; 套一层本容器后即可滚动查看全部内容.
    注意: 内容控件需 pack_propagate(True), 否则高度被压成 1px 导致无法滚动.
    """
    def __init__(self, parent, width=380, bg='#f0f0f0', **kw):
        super().__init__(parent, **kw)
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0, bg=bg)
        self.vbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set, yscrollincrement=20)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.vbar.pack(side='right', fill='y')

        self.body = tk.Frame(self.canvas, bg=bg)
        self._body_id = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.body.bind(
            '<Configure>',
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind(
            '<Configure>',
            lambda e: self.canvas.itemconfigure(self._body_id, width=e.width))

        # 滚轮: 绑到 all, 但只在指针位于本容器内时才滚动 (不影响绘图区/对话框)
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)

    def _on_mousewheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self:
            if widget is self.canvas:
                box = self.canvas.bbox('all')
                if box and box[3] > self.canvas.winfo_height():   # 内容超高才滚
                    steps = int(-event.delta / 120) or (-1 if event.delta > 0 else 1)
                    self.canvas.yview_scroll(steps, 'units')
                return
            widget = widget.master

    def scroll_to_widget(self, widget):
        """把指定控件滚动到可见区域 (调试/跳转用)."""
        self.canvas.update_idletasks()
        box = self.canvas.bbox('all')
        if not box or box[3] <= self.canvas.winfo_height():
            return
        self.canvas.yview_moveto(max(0.0, min(1.0, widget.winfo_y() / box[3])))


class ControlPanel(tk.Frame):
    def __init__(self, parent, config, cbs):
        super().__init__(parent, bg='#f0f0f0')
        self.config = config
        self.cbs = cbs
        # 宽度/高度由内容决定: 由 App 侧的 ScrollFrame 承载滚动, 不再锁死 320px

        # --- 标题 ---
        tk.Label(self, text="Radar V22 (Fixed)", bg='#f0f0f0', font=('Arial', 12, 'bold')).pack(pady=5)

        # --- 模式选择 ---
        frm_mode = tk.Frame(self, bg='#f0f0f0')
        frm_mode.pack(fill='x', padx=5, pady=5)
        self.mode_var = tk.StringVar(value=config.connection_mode)
        ttk.Combobox(frm_mode, textvariable=self.mode_var, values=('UDP', 'BD_UDP', 'BD_CAN', 'SERIAL', 'CAN', 'PLAYBACK'), state='readonly').pack(fill='x')
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

        # --- CAN 配置面板 (仅 CAN 模式显示) ---
        self.frm_can = tk.LabelFrame(self, text="CAN Settings", bg='#f0f0f0', fg='#c60')
        self._entry(self.frm_can, "CAN Device:", "can_device_type")
        self._entry(self.frm_can, "MCU Name:", "mcu_name")
        self._entry(self.frm_can, "Packet Size:", "can_packet_size")
        self._entry(self.frm_can, "Header Size:", "can_header_size")
        self._entry(self.frm_can, "CIR Data Size:", "can_cir_data_size")
        self._entry(self.frm_can, "UCI Signature:", "can_uci_signature")
        self._entry(self.frm_can, "Packet Count:", "can_packet_cnt")

        # --- 算法设置 ---
        frm_algo = tk.LabelFrame(self, text="Algo & Geometry", bg='#f0f0f0', fg='purple')
        frm_algo.pack(fill='x', padx=5, pady=5)

        row1 = tk.Frame(frm_algo, bg='#f0f0f0')
        row1.pack(fill='x', padx=2)
        tk.Label(row1, text="Algo:", bg='#f0f0f0').pack(side='left')
        self.cb_algo = ttk.Combobox(row1, values=('PLOT', '2D-MUSIC', 'POINT-CLOUD', 'POINT-CLOUD-OPTIMIZED', 'POINT-CLOUD-DUBHE', 'POINT-CLOUD-PAPER', 'RA-CFAR', 'RA-HEATMAP', 'RA-OCCUPANCY', 'ANGLE-SPECTRUM', 'AS-RAW'), width=22, state='readonly')
        self.cb_algo.set(config.current_algo)
        self.cb_algo.pack(side='left')
        self.cb_algo.bind("<<ComboboxSelected>>", self._on_algo_change)
        tk.Button(row1, text="⚙️ Params", command=self._open_algo).pack(side='right')
        tk.Button(row1, text="🪑 Seats", command=self._open_seats).pack(side='right', padx=(2, 5))

        row2 = tk.Frame(frm_algo, bg='#f0f0f0')
        row2.pack(fill='x', padx=2)
        tk.Label(row2, text="Layout:", bg='#f0f0f0').pack(side='left')
        self.cb_layout = ttk.Combobox(row2, values=list(ANTENNA_LAYOUTS.keys()), state='readonly')
        self.cb_layout.set(config.current_layout_name)
        self.cb_layout.pack(side='right', fill='x', expand=True)
        self.cb_layout.bind("<<ComboboxSelected>>", self._on_layout_change)

        # --- RA-OCCUPANCY 显示模式: 热力图 / 定位图(仅峰值红点) ---
        self.frm_ra_occ_view = tk.Frame(frm_algo, bg='#f0f0f0')
        self.var_ra_occ_loc = tk.BooleanVar(
            value=bool(config.algo_params.get('RA-OCCUPANCY', {})
                       .get('ra_occ_show_localization', False)))
        tk.Checkbutton(self.frm_ra_occ_view, text="定位图 (仅峰值红点)",
                       variable=self.var_ra_occ_loc, bg='#f0f0f0',
                       command=self._on_ra_occ_view_toggle).pack(side='left')
        self._refresh_ra_occ_view_row()

        # --- [重构] 录制设置面板: 座位勾选 + 目录下拉 + 文件名自动生成 ---
        frm_rec = tk.LabelFrame(self, text="Recording Settings", bg='#f0f0f0', fg='blue')
        frm_rec.pack(fill='x', padx=5, pady=5)

        # 1. 占用状态勾选
        self._updating_filename = False
        self._record_speech_jobs = []
        self.var_io_mode = tk.StringVar(value="in")
        self.var_person_id = tk.StringVar(value="a1")
        self.var_record_area = tk.StringVar(value="d")
        self.var_record_pos = tk.StringVar(value="1")
        self.var_record_pose = tk.StringVar(value="s1")
        self.var_out_actions = tk.StringVar(value="")
        self.var_out_pos_count = tk.StringVar(value="1")
        self.var_out_action_time = tk.StringVar(value="10")

        row_io = tk.Frame(frm_rec, bg='#f0f0f0')
        row_io.pack(fill='x', padx=2, pady=(5, 2))
        tk.Label(row_io, text="采样类型:", bg='#f0f0f0', width=12, anchor='w').pack(side='left')
        for text, value in (("in", "in"), ("out", "out")):
            tk.Radiobutton(row_io, text=text, value=value, variable=self.var_io_mode,
                           bg='#f0f0f0', command=self._on_io_mode_change).pack(side='left', padx=3)

        row_person = tk.Frame(frm_rec, bg='#f0f0f0')
        row_person.pack(fill='x', padx=2, pady=2)
        tk.Label(row_person, text="人员编号:", bg='#f0f0f0', width=12, anchor='w').pack(side='left')
        tk.Entry(row_person, textvariable=self.var_person_id).pack(side='left', fill='x', expand=True)

        row_area = tk.Frame(frm_rec, bg='#f0f0f0')
        row_area.pack(fill='x', padx=2, pady=2)
        tk.Label(row_area, text="脚坑/座位:", bg='#f0f0f0', width=12, anchor='w').pack(side='left')
        for text, value in (("脚坑", "d"), ("座位", "o")):
            tk.Radiobutton(row_area, text=text, value=value, variable=self.var_record_area,
                           bg='#f0f0f0', command=self._refresh_filename_preview).pack(side='left', padx=3)

        row_pos = tk.Frame(frm_rec, bg='#f0f0f0')
        row_pos.pack(fill='x', padx=2, pady=2)
        tk.Label(row_pos, text="位置编号:", bg='#f0f0f0', width=12, anchor='w').pack(side='left')
        for value in ("1", "2", "3", "4"):
            tk.Radiobutton(row_pos, text=value, value=value, variable=self.var_record_pos,
                           bg='#f0f0f0', command=self._refresh_filename_preview).pack(side='left', padx=3)

        row_pose_sit = tk.Frame(frm_rec, bg='#f0f0f0')
        row_pose_sit.pack(fill='x', padx=2, pady=2)
        tk.Label(row_pose_sit, text="姿势-坐:", bg='#f0f0f0', width=12, anchor='w').pack(side='left')
        for text, value in (("坐1", "s1"), ("坐2", "s2"), ("坐3", "s3")):
            tk.Radiobutton(row_pose_sit, text=text, value=value, variable=self.var_record_pose,
                           bg='#f0f0f0', command=self._refresh_filename_preview).pack(side='left', padx=3)

        row_pose_lie = tk.Frame(frm_rec, bg='#f0f0f0')
        row_pose_lie.pack(fill='x', padx=2, pady=2)
        tk.Label(row_pose_lie, text="姿势-躺:", bg='#f0f0f0', width=12, anchor='w').pack(side='left')
        for text, value in (("躺1", "l1"), ("躺2", "l2"), ("躺3", "l3"), ("躺4", "l4")):
            tk.Radiobutton(row_pose_lie, text=text, value=value, variable=self.var_record_pose,
                           bg='#f0f0f0', command=self._refresh_filename_preview).pack(side='left', padx=3)

        row_out_actions = tk.Frame(frm_rec, bg='#f0f0f0')
        row_out_actions.pack(fill='x', padx=2, pady=2)
        tk.Label(row_out_actions, text="动作:", bg='#f0f0f0', width=12, anchor='w').pack(side='left')
        tk.Entry(row_out_actions, textvariable=self.var_out_actions).pack(side='left', fill='x', expand=True)

        row_out_pos = tk.Frame(frm_rec, bg='#f0f0f0')
        row_out_pos.pack(fill='x', padx=2, pady=2)
        tk.Label(row_out_pos, text="位置数量:", bg='#f0f0f0', width=12, anchor='w').pack(side='left')
        tk.Entry(row_out_pos, textvariable=self.var_out_pos_count, width=8).pack(side='left')

        row_out_time = tk.Frame(frm_rec, bg='#f0f0f0')
        row_out_time.pack(fill='x', padx=2, pady=2)
        tk.Label(row_out_time, text="单动作时间:", bg='#f0f0f0', width=12, anchor='w').pack(side='left')
        tk.Entry(row_out_time, textvariable=self.var_out_action_time, width=8).pack(side='left')
        tk.Label(row_out_time, text="秒", bg='#f0f0f0', fg='gray').pack(side='left', padx=4)
        self.lbl_out_total_duration = tk.Label(row_out_time, text="", bg='#f0f0f0', fg='gray')
        self.lbl_out_total_duration.pack(side='left', padx=4)

        self.var_person_id.trace_add("write", lambda *args: self._refresh_filename_preview())
        self.var_out_actions.trace_add("write", lambda *args: self._on_out_record_config_change())
        self.var_out_pos_count.trace_add("write", lambda *args: self._on_out_record_config_change())
        self.var_out_action_time.trace_add("write", lambda *args: self._on_out_record_config_change())
        self._in_record_rows = [row_area, row_pos, row_pose_sit, row_pose_lie]
        self._out_record_rows = [row_out_actions, row_out_pos, row_out_time]
        for row in self._out_record_rows:
            row.pack_forget()

        # 2. 保存目录下拉 + 浏览
        row_dir = tk.Frame(frm_rec, bg='#f0f0f0')
        self.row_save_dir = row_dir
        row_dir.pack(fill='x', padx=2, pady=2)
        tk.Label(row_dir, text="保存目录:", bg='#f0f0f0', anchor='w').pack(side='left')
        # 确保默认目录在列表里
        if config.data_save_dir not in config.recent_save_dirs:
            config.recent_save_dirs.insert(0, config.data_save_dir)
        self._recent_dirs = config.recent_save_dirs
        self.var_save_dir = tk.StringVar(value=config.data_save_dir)
        self.cb_save_dir = ttk.Combobox(row_dir, textvariable=self.var_save_dir,
                                        values=self._recent_dirs, width=28)
        self.cb_save_dir.pack(side='left', fill='x', expand=True, padx=(0, 2))
        self.cb_save_dir.bind('<<ComboboxSelected>>', self._on_save_dir_change)
        self.cb_save_dir.bind('<FocusOut>', self._on_save_dir_change)
        tk.Button(row_dir, text="浏览...", width=6, font=('Arial', 8),
                  command=self._choose_dir).pack(side='right')

        # 3. 文件名预览
        row_fname = tk.Frame(frm_rec, bg='#f0f0f0')
        row_fname.pack(fill='x', padx=2, pady=2)
        tk.Label(row_fname, text="文件名:", bg='#f0f0f0', anchor='w').pack(side='left')
        self.var_record_filename = tk.StringVar(value="")
        self.entry_record_filename = tk.Entry(row_fname, textvariable=self.var_record_filename,
                                              font=('Arial', 8), relief='sunken')
        self.entry_record_filename.pack(side='left', fill='x', expand=True)
        self.var_record_filename.trace_add("write", lambda *args: self._on_record_filename_edit())
        self._refresh_filename_preview()

        # 4. 时长设置
        row_dur = tk.Frame(frm_rec, bg='#f0f0f0')
        self.row_rec_duration = row_dur
        row_dur.pack(fill='x', padx=2, pady=2)
        self.lbl_rec_duration = tk.Label(row_dur, text="录制时长:", bg='#f0f0f0', anchor='w')
        self.lbl_rec_duration.pack(side='left')
        self.var_rec_dur = tk.DoubleVar(value=config.record_duration)
        tk.Entry(row_dur, textvariable=self.var_rec_dur, width=6).pack(side='left', padx=4)
        tk.Label(row_dur, text="秒 (0=不限)", bg='#f0f0f0', fg='gray').pack(side='left')

        # 5. 录制按钮
        self.btn_rec = tk.Button(frm_rec, text="Start Recording", command=self._rec, bg='#ddd')
        self.btn_rec.pack(fill='x', padx=5, pady=5)
        self._on_io_mode_change()

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

    def _pack_record_rows(self, rows, visible):
        for row in rows:
            if visible:
                row.pack(fill='x', padx=2, pady=2, before=self.row_save_dir)
            else:
                row.pack_forget()

    def _on_io_mode_change(self):
        is_out = (self.var_io_mode.get() == 'out')
        self._pack_record_rows(self._in_record_rows, not is_out)
        self._pack_record_rows(self._out_record_rows, is_out)
        if is_out:
            self.row_rec_duration.pack_forget()
            self._update_out_total_duration()
        else:
            self.row_rec_duration.pack(fill='x', padx=2, pady=2, before=self.btn_rec)
        self._refresh_filename_preview()

    def _parse_out_actions(self):
        raw = self.var_out_actions.get().replace('，', ',')
        return [item.strip() for item in raw.split(',') if item.strip()]

    def _get_out_position_count(self):
        count = int(self.var_out_pos_count.get())
        if count <= 0:
            raise ValueError("position count must be positive")
        return count

    def _get_out_action_time(self):
        seconds = float(self.var_out_action_time.get())
        if seconds <= 0:
            raise ValueError("action time must be positive")
        return seconds

    def _calc_out_record_duration(self):
        actions = self._parse_out_actions()
        if not actions:
            raise ValueError("actions required")
        return self._get_out_action_time() * len(actions) * self._get_out_position_count()

    def _update_out_total_duration(self):
        try:
            total = self._calc_out_record_duration()
            self.var_rec_dur.set(total)
            self.lbl_out_total_duration.config(text=f"总时长: {total:.1f} 秒")
        except Exception:
            self.lbl_out_total_duration.config(text="总时长: --")

    def _on_out_record_config_change(self):
        if self.var_io_mode.get() == 'out':
            self._update_out_total_duration()

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

    # ===== 录制文件名自动生成辅助方法 =====
    def _gen_base_filename(self):
        """根据采样标签生成基础文件名."""
        person_id = ''.join(ch for ch in self.var_person_id.get().strip() if ch.isalnum())
        if not person_id:
            person_id = "a1"
        if self.var_io_mode.get() == 'out':
            return f"out_{person_id}_1.bin"
        area_pos = f"{self.var_record_area.get()}{self.var_record_pos.get()}"
        pose = self.var_record_pose.get()
        return f"in_{person_id}_{area_pos}_{pose}_1.bin"

    def _normalize_record_filename(self, filename):
        filename = filename.strip()
        if not filename:
            filename = self._gen_base_filename()
        filename = os.path.basename(filename)
        if not filename.lower().endswith('.bin'):
            filename += '.bin'
        return filename

    def _get_unique_filepath(self, directory, base_name):
        """在 directory 下找不冲突的文件名, 必要时加 _2, _3... 后缀.
        返回 (完整路径, 实际使用的文件名)."""
        base_stem, ext = os.path.splitext(base_name)
        if not ext:
            ext = '.bin'
        prefix = base_stem
        start_index = 1
        if '_' in base_stem:
            maybe_prefix, maybe_index = base_stem.rsplit('_', 1)
            if maybe_index.isdigit():
                prefix = maybe_prefix
                start_index = max(1, int(maybe_index))
        n = start_index
        while True:
            new_name = f"{prefix}_{n}{ext}"
            candidate = os.path.join(directory, new_name)
            if not os.path.exists(candidate):
                return candidate, new_name
            n += 1

    def _refresh_filename_preview(self):
        """更新文件名输入框."""
        base = self._gen_base_filename()
        directory = self.var_save_dir.get().strip() or self.config.data_save_dir
        full_path, actual_name = self._get_unique_filepath(directory, base)
        self._updating_filename = True
        self.var_record_filename.set(actual_name)
        self._updating_filename = False
        self._cached_rec_path = full_path

    def _update_cached_record_path_from_filename(self):
        filename = self._normalize_record_filename(self.var_record_filename.get())
        directory = self.var_save_dir.get().strip() or self.config.data_save_dir
        self._cached_rec_path = os.path.join(directory, filename)
        self._updating_filename = True
        self.var_record_filename.set(filename)
        self._updating_filename = False

    def _on_record_filename_edit(self):
        if self._updating_filename:
            return
        self._update_cached_record_path_from_filename()

    def _on_occ_checkbox_change(self):
        self._refresh_filename_preview()

    def _on_save_dir_change(self, *_):
        directory = self.var_save_dir.get().strip()
        if directory and directory not in self._recent_dirs:
            self._recent_dirs.insert(0, directory)
            if len(self._recent_dirs) > 10:
                self._recent_dirs = self._recent_dirs[:10]
            self.config.recent_save_dirs = self._recent_dirs
            self.cb_save_dir['values'] = self._recent_dirs
        self.config.data_save_dir = directory
        self._update_cached_record_path_from_filename()
    def _occ_select_all(self):
        self._refresh_filename_preview()

    def _occ_clear_all(self):
        self._refresh_filename_preview()

    def _occ_select_front(self, num_seats):
        """前排: 前2个座椅选中, 其余清空 (模拟常见的前排有人场景)"""
        self._refresh_filename_preview()

    def _choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.var_save_dir.get() or self.config.data_save_dir)
        if d:
            self.config.data_save_dir = d
            self.var_save_dir.set(d)
            self._on_save_dir_change()

    def _on_mode_change(self, *_):
        self.config.connection_mode = self.mode_var.get()
        is_pb = (self.config.connection_mode == 'PLAYBACK')
        is_can = (self.config.connection_mode in ('CAN', 'BD_CAN'))

        if is_pb:
            self.btn_rec.config(state='disabled')
            self.frm_pb.pack(fill='x', padx=5, pady=5)
            self.pb_bar['value'] = 0
            self.lbl_prog_text.config(text="Progress: 0.0%")
        else:
            self.btn_rec.config(state='normal')
            self.frm_pb.pack_forget()

        # CAN 配置面板显隐
        if is_can:
            self.frm_can.pack(fill='x', padx=5, pady=2)
        else:
            self.frm_can.pack_forget()

    def _on_layout_change(self, e): 
        self.config.load_layout(self.cb_layout.get())

    def _on_algo_change(self, e): 
        self.config.current_algo = self.cb_algo.get()
        self._refresh_ra_occ_view_row()
        self.cbs['update_layout'](self.config.current_algo)

    def _on_ra_occ_view_toggle(self):
        """勾选/取消定位图: 只写参数并实时切换显示, 不触发 update_layout."""
        self.cbs['ra_occ_view'](self.var_ra_occ_loc.get())

    def _refresh_ra_occ_view_row(self):
        """仅当前算法为 RA-OCCUPANCY 时显示该开关, 并与参数保持同步."""
        if self.config.current_algo == 'RA-OCCUPANCY':
            self.var_ra_occ_loc.set(bool(
                self.config.algo_params.get('RA-OCCUPANCY', {})
                .get('ra_occ_show_localization', False)))
            self.frm_ra_occ_view.pack(fill='x', padx=2, pady=(3, 0))
        else:
            self.frm_ra_occ_view.pack_forget()

    def _open_algo(self):
        AlgoSettingsDialog(self, self.config.current_algo,
                           self.config.algo_params.get(self.config.current_algo, {}),
                           lambda p: (self.config.algo_params.update({self.config.current_algo: p}),
                                      save_config(self.config),
                                      self.cbs['update_layout'](self.config.current_algo),
                                      self._refresh_ra_occ_view_row()))

    def _refresh_occ_checkboxes(self):
        """录制命名已改为固定采样标签, 座椅配置变化时只刷新文件名."""
        self._refresh_filename_preview()

    def _open_seats(self):
        occ_params = self.config.algo_params.get('SEAT-OCCUPANCY', {})
        def on_save(new_p):
            self.config.algo_params['SEAT-OCCUPANCY'] = new_p
            save_config(self.config)
            # 重建检测器 + 刷新当前布局使椭圆立即生效
            self.cbs['update_layout'](self.config.current_algo)
            # 刷新录制面板的座位勾选框（支持4/5座切换）
            self._refresh_occ_checkboxes()
        SeatConfigDialog(self, occ_params, on_save)

    def _sel_pb(self):
        # 修改为 askopenfilenames (复数)
        files = filedialog.askopenfilenames(filetypes=[("Radar Bin", "*.bin"), ("All", "*.*")])
        if files:
            # 将选中的文件列表存入 config (需确保 config 有这个属性)
            self.config.playback_file_list = list(files)
            self.config.playback_file = files[0] # 默认第一个
            count = len(files)
            self.lbl_pb.config(text=f"Selected {count} files") # UI 显示数量

    def _speak_async(self, text):
        if not text:
            return
        escaped = text.replace("'", "''")
        cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{escaped}')"
        )
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[Speech] 播报失败: {e}")

    def _schedule_record_speech(self, delay_ms, text):
        job = self.after(max(0, int(delay_ms)), lambda: self._speak_async(text))
        self._record_speech_jobs.append(job)

    def _clear_record_speech(self):
        for job in self._record_speech_jobs:
            try:
                self.after_cancel(job)
            except Exception:
                pass
        self._record_speech_jobs = []

    def _start_record_speech(self, duration):
        self._clear_record_speech()
        self._schedule_record_speech(0, "开始采样")
        if self.var_io_mode.get() == 'out':
            actions = self._parse_out_actions()
            action_time = self._get_out_action_time()
            pos_count = self._get_out_position_count()
            for pos_idx in range(pos_count):
                for action_idx, action in enumerate(actions):
                    elapsed = (pos_idx * len(actions) + action_idx) * action_time
                    if pos_idx > 0 and action_idx == 0:
                        self._schedule_record_speech(elapsed * 1000, "更换位置")
                        self._schedule_record_speech(elapsed * 1000 + 1000, action)
                    elif pos_idx == 0 and action_idx == 0:
                        self._schedule_record_speech(1000, action)
                    else:
                        self._schedule_record_speech(elapsed * 1000, action)
        if duration > 0:
            self._schedule_record_speech(duration * 1000, "结束采样")

    def _rec(self):
        if not self.recording_state:
            # --- 开始录制 ---
            try:
                # 1. 获取目录并自动生成文件名
                save_dir = self.var_save_dir.get().strip()
                self.config.data_save_dir = save_dir

                # 使用当前文件名输入框内容, 避免覆盖手动修改
                self._update_cached_record_path_from_filename()
                full_path = self._cached_rec_path
                fname = os.path.basename(full_path)

                self.config.record_filename = fname

                if self.var_io_mode.get() == 'out':
                    self.config.record_duration = self._calc_out_record_duration()
                    self.var_rec_dur.set(self.config.record_duration)
                else:
                    self.config.record_duration = float(self.var_rec_dur.get())
            except ValueError:
                messagebox.showerror("Error", "无效的参数输入")
                return

            # 2. 检查并创建目录
            if not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir)
                except:
                    messagebox.showerror("Error", "无法创建保存目录")
                    return

            # 3. 调用后端开始录制
            if self.cbs['rec_start'](full_path, self.config.record_duration):
                self.recording_state = True
                self.btn_rec.config(bg='#f88', text="Stop Recording")
                self._start_record_speech(self.config.record_duration)
        else:
            # --- 停止录制 ---
            self.cbs['rec_stop']()
            self._clear_record_speech()
            self._speak_async("结束采样")
            self.recording_state = False
            self.btn_rec.config(bg='#ddd', text="Start Recording")
            self._refresh_filename_preview()

    def update_ui(self, run, rec, rec_time, fps, pb_prog):
        # 系统状态
        self.lbl_status.config(text=f"Running ({fps:.1f} FPS)" if run else "Stopped", bg='#8f8' if run else '#ccc')
        self.btn_start.config(state='disabled' if run else 'normal')
        self.btn_stop.config(state='normal' if run else 'disabled')
        
        # 录制按钮状态逻辑
        if self.config.connection_mode != 'PLAYBACK':
            # 如果系统未运行，通常不允许录制，或者允许录制空数据？通常是不允许
            self.btn_rec.config(state='normal' if run else 'disabled')
            was_recording = self.recording_state
            
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
                if was_recording:
                    self._clear_record_speech()
                    self._refresh_filename_preview()
        
        # --- [新增] 更新进度条逻辑 ---
        if self.config.connection_mode == 'PLAYBACK':
            # 更新进度条数值 (0-100)
            self.pb_bar['value'] = pb_prog
            # 更新百分比文字显示
            self.lbl_prog_text.config(text=f"Progress: {pb_prog:.1f}%")
    # ----------------------------

class App:
    def __init__(self, root):
        self.root = root; fit_to_screen(root); self.config = RadarConfig(); self.config.load_layout("2x4_Default")
        load_config(self.config)  # 持久化: 用已保存的配置覆盖默认值
        if not os.path.exists(self.config.data_save_dir): os.makedirs(self.config.data_save_dir)
        self.data_manager = RadarDataManager(self.config); self.algo_processor = AlgorithmProcessor(self.config, self.data_manager)
        self.source = None; self.running = False
        self.root.columnconfigure(1, weight=1); self.root.rowconfigure(0, weight=1)
        cbs = {'start': self.start, 'stop': self.stop, 'rec_start': self.rec_start, 'rec_stop': self.rec_stop, 'update_layout': self.update_layout, 'ra_occ_view': self.set_ra_occ_view}
        # 左栏放进可滚动容器: 小屏/高 DPI 下也能访问全部控件(含底部 START/STOP)
        self.ctrl_host = ScrollFrame(root); self.ctrl_host.grid(row=0, column=0, sticky='ns')
        self.ctrl = ControlPanel(self.ctrl_host.body, self.config, cbs)
        self.ctrl.pack_propagate(True)      # 让面板按内容撑开(否则高度被压成 1px, 滚不动)
        self.ctrl.pack(fill='x')
        self.plot_panel = PlotPanel(root); self.plot_panel.grid(row=0, column=1, sticky='nsew')
        self.plot_panel.init_layout("PLOT", self.config.algo_params['PLOT'])
        self._ra_occ_snapshot_counter = 0
        self._last_ra_occ_heatmap_snapshot = None
        self._last_hm_run_snapshot = -1      # [加速] RA-HEATMAP 上次产图对应的快照号
        self._ra_occ_h_bg_buffer = deque(maxlen=8)
        self._ra_occ_h_buffer = deque(maxlen=8)
        self._ra_occ_raw_window_buffer = deque(maxlen=8)
        self._bd_bg_removed_snapshot_buffer = deque(maxlen=3000)
        self.bd_trigger_count = 0
        self._ra_occ_interpreter = None
        self._ra_occ_model_path = None
        self._ra_occ_model_failed = False
        self._ra_occ_playback_sample_index = 0
        self.pc_export_data = []  # 新增：用于存储点云导出数据的列表
        self.playback_queue = [] # 新增：存放待处理的文件路径队列
        self.playback_skip_state = False  # False 表示处理，True 表示跳过
        # 座椅占用检测器
        occ_params = self.config.algo_params.get('SEAT-OCCUPANCY', {})
        self.seat_detector = SeatOccupancyDetector(occ_params)
        self.ra_occupancy_detector = RAOccupancyDetector(occ_params)
    def set_ra_occ_view(self, localization):
        """RA-OCCUPANCY 显示模式: False=热力图 / True=定位图(仅峰值红点)."""
        params = self.config.algo_params.setdefault('RA-OCCUPANCY', {})
        params['ra_occ_show_localization'] = bool(localization)
        save_config(self.config)                       # 立即持久化
        self.plot_panel.set_ra_occ_view(bool(localization))

    def update_layout(self, mode):
        # 1. 更新绘图面板的布局
        p = self.config.algo_params.get(mode, {})
        occ_cfg = self.config.algo_params.get('SEAT-OCCUPANCY', {})
        p['occupancy_config'] = occ_cfg
        self.plot_panel.init_layout(mode, p)

        # 2. 重建座椅检测器 (使配置修改立即生效)
        self.seat_detector = SeatOccupancyDetector(occ_cfg)
        self.ra_occupancy_detector = RAOccupancyDetector(occ_cfg)

        # 3. 【核心修复】强制算法处理器重新初始化参数
        # 这样当你修改了虚拟天线索引、频率范围等参数时，后端才会重新计算
        self.algo_processor.init_done = False 
        if mode == 'RA-OCCUPANCY':
            self._last_ra_occ_heatmap_snapshot = None
            self._ra_occ_h_bg_buffer.clear()
            self._ra_occ_h_buffer.clear()
            self._ra_occ_raw_window_buffer.clear()
            self._bd_bg_removed_snapshot_buffer.clear()
        print(f"参数已更新，算法 {mode} 将重新初始化...")
    def start(self):
        RadarProtocol.update_protocol(self.config.ft_len)
        if self.config.current_algo == 'RA-OCCUPANCY':
            self.plot_panel.ra_occ_oa_filter_history.clear()
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
            if self.config.connection_mode in ('BD_UDP', 'BD_CAN'):
                self.bd_trigger_count = 0
            self._ra_occ_snapshot_counter = 0
            self._last_ra_occ_heatmap_snapshot = None
            self._ra_occ_h_bg_buffer.clear()
            self._ra_occ_h_buffer.clear()
            self._ra_occ_raw_window_buffer.clear()
            self._bd_bg_removed_snapshot_buffer.clear()
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
        
        # 3. 重置导出缓存、UI 和座椅检测器
        self.pc_export_data = []
        self.plot_panel.pc_history = []
        occ_params = self.config.algo_params.get('SEAT-OCCUPANCY', {})
        self.seat_detector = SeatOccupancyDetector(occ_params)
        self.ra_occupancy_detector = RAOccupancyDetector(occ_params)
        self._ra_occ_snapshot_counter = 0
        self.plot_panel.ra_occ_oa_filter_history.clear()
        self._last_ra_occ_heatmap_snapshot = None
        self._ra_occ_playback_sample_index = 0
        self._ra_occ_h_bg_buffer.clear()
        self._ra_occ_h_buffer.clear()
        self._ra_occ_raw_window_buffer.clear()
        self._bd_bg_removed_snapshot_buffer.clear()
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
        save_config(self.config)  # 持久化: 退出前保存配置
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

    def _save_matrix_sample_txt(self, matrix, path):
        matrix = np.asarray(matrix)
        with open(path, "w", encoding="utf-8") as f:
            for idx in range(matrix.shape[0]):
                np.savetxt(f, matrix[idx], fmt="%.10e")

    def _bd_sample_folder_name(self, sample):
        p = self.config.algo_params.get('RA-OCCUPANCY', {})
        stride = max(1, int(p.get('heatmap_update_stride_combined', 4)))
        path_count = int(np.asarray(sample).shape[2]) if np.asarray(sample).ndim >= 3 else 0
        azimuth_num = int(p.get('azimuth_num', np.asarray(sample).shape[0] if np.asarray(sample).ndim >= 1 else 0))
        keep_range = p.get('range_bin_keep_range')
        if keep_range is not None and len(keep_range) == 2:
            range_part = f"{int(keep_range[0])}-{int(keep_range[1])}"
        else:
            range_part = f"bins={np.asarray(sample).shape[1] if np.asarray(sample).ndim >= 2 else 0}"
        return f"combined={stride}_range={range_part}_path={path_count}_azimuth={azimuth_num}"

    def _bd_pack_background_removed_frame(self, raw_frame, cir_no_bg):
        if raw_frame is None or cir_no_bg is None:
            return None
        if len(raw_frame) != RadarProtocol.FRAME_LEN:
            return None

        cir_no_bg = np.asarray(cir_no_bg)
        if cir_no_bg.size != RadarProtocol.FT_LEN:
            return None

        iq = np.empty(cir_no_bg.size * 2, dtype=np.int16)
        iq[0::2] = np.clip(np.rint(cir_no_bg.real), -32768, 32767).astype(np.int16)
        iq[1::2] = np.clip(np.rint(cir_no_bg.imag), -32768, 32767).astype(np.int16)
        payload_start = RadarProtocol.HEADER_LEN + RadarProtocol.ANTENNA_INFO_LEN
        payload_end = payload_start + RadarProtocol.CIR_DATA_LEN
        return raw_frame[:payload_start] + iq.tobytes() + raw_frame[payload_end:]

    def _bd_remember_bg_removed_frame(self, snapshot_idx, tx, rx, raw_frame, cir_no_bg):
        bg_removed_frame = self._bd_pack_background_removed_frame(raw_frame, cir_no_bg)
        if bg_removed_frame is None or snapshot_idx <= 0:
            return
        if not self._bd_bg_removed_snapshot_buffer or self._bd_bg_removed_snapshot_buffer[-1]['idx'] != snapshot_idx:
            self._bd_bg_removed_snapshot_buffer.append({'idx': snapshot_idx, 'frames': []})
        self._bd_bg_removed_snapshot_buffer[-1]['frames'].append(bg_removed_frame)

    def _remember_ra_occ_raw_window(self):
        p = self.config.algo_params.get('RA-OCCUPANCY', {})
        snapshot_end = self._last_ra_occ_heatmap_snapshot
        if snapshot_end is None:
            return
        snapshots = max(1, int(p.get('snapshots', self.config.max_snapshots)))
        snapshot_start = max(1, snapshot_end - snapshots + 1)
        self._ra_occ_raw_window_buffer.append((snapshot_start, snapshot_end))

    def _bd_bg_removed_frames_for_current_sample(self):
        if not self._ra_occ_raw_window_buffer:
            return [], None, None

        snapshot_start = min(win[0] for win in self._ra_occ_raw_window_buffer)
        snapshot_end = max(win[1] for win in self._ra_occ_raw_window_buffer)
        frames = []
        for item in self._bd_bg_removed_snapshot_buffer:
            idx = item['idx']
            if snapshot_start <= idx <= snapshot_end:
                frames.extend(item['frames'])
        return frames, snapshot_start, snapshot_end

    def _save_bd_bg_removed_signal_bin(self, path, sample, score=None):
        frames, snapshot_start, snapshot_end = self._bd_bg_removed_frames_for_current_sample()
        mode = self.config.connection_mode
        if not frames:
            print(f"[{mode}] background-removed signal save skipped: snapshot history is empty")
            return 0

        with open(path, "wb") as f:
            for frame in frames:
                f.write(frame)

        meta = asdict(self.config)
        meta.update({
            "recorded_date": str(datetime.now()),
            "bd_event_score": score,
            "bd_background_removed_frame_count": len(frames),
            "bd_snapshot_start": snapshot_start,
            "bd_snapshot_end": snapshot_end,
            "bd_source": "background_removed_snapshot_window",
            "bd_payload": "remove_background_complex_int16_iq",
        })
        try:
            with open(path + ".meta", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, cls=NpEncoder, ensure_ascii=False)
        except Exception as e:
            print(f"[{mode}] background-removed signal meta save failed: {e}")

        print(f"[{mode}] saved background-removed signal: {path} ({len(frames)} frames)")
        return len(frames)

    def _save_bd_positive_sample(self, sample, score=None):
        self._count_bd_trigger(score)

    def _count_bd_trigger(self, score=None):
        self.bd_trigger_count += 1
        score_text = "none" if score is None else f"{float(score):.4f}"
        print(f"[{self.config.connection_mode}] positive trigger #{self.bd_trigger_count}: score={score_text}")

    def _ra_occ_normalize_h(self, h, eps=1e-6):
        mean = np.mean(h)
        std = np.std(h)
        if std <= eps:
            return h
        return (h - mean) / std

    def _ra_occ_model_enabled(self):
        value = self.config.algo_params.get('RA-OCCUPANCY', {}).get('sample_enable', True)
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on', 'enable', 'enabled')
        return bool(value)

    def _export_ra_occ_playback_sample(self, sample):
        """回放模式下导出送入模型前的 [angle, range, time] 标准化样本。"""
        if self.config.connection_mode != 'PLAYBACK':
            return

        p = self.config.algo_params.get('RA-OCCUPANCY', {})
        try:
            enabled = int(p.get('playback_sample_export', 0)) == 1
        except (TypeError, ValueError):
            enabled = False
        if not enabled or not self.config.playback_file:
            return

        sample = np.asarray(sample)
        playback_name = os.path.splitext(os.path.basename(self.config.playback_file))[0]
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), playback_name)
        try:
            os.makedirs(output_dir, exist_ok=True)
            while True:
                output_path = os.path.join(
                    output_dir, f"{playback_name}_{self._ra_occ_playback_sample_index}.txt")
                if not os.path.exists(output_path):
                    break
                self._ra_occ_playback_sample_index += 1

            self._save_matrix_sample_txt(sample, output_path)

            self._ra_occ_playback_sample_index += 1
            print(f"[RA-OCCUPANCY] exported playback sample: {output_path}")
        except (OSError, ValueError) as e:
            print(f"[RA-OCCUPANCY] playback sample export failed: {e}")

    def _ra_occ_resolve_model_path(self, model_path):
        if not model_path:
            model_path = "./model/epoch-25-val-f1-100.0-sp-100.0.tflite"
        if os.path.isabs(model_path):
            return model_path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(base_dir, model_path))

    def _ra_occ_get_interpreter(self):
        p = self.config.algo_params.get('RA-OCCUPANCY', {})
        model_path = self._ra_occ_resolve_model_path(p.get('oa_model_path', ''))
        if self._ra_occ_interpreter is not None and self._ra_occ_model_path == model_path:
            return self._ra_occ_interpreter
        if self._ra_occ_model_failed and self._ra_occ_model_path == model_path:
            return None

        self._ra_occ_model_path = model_path
        self._ra_occ_interpreter = None
        self._ra_occ_model_failed = False
        try:
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=model_path)
            except ImportError:
                from tflite_runtime.interpreter import Interpreter
                interpreter = Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            self._ra_occ_interpreter = interpreter
            return interpreter
        except Exception as e:
            print(f"[RA-OCCUPANCY] model load failed: {e}")
            self._ra_occ_model_failed = True
            return None

    def _ra_occ_run_model(self, sample):
        interpreter = self._ra_occ_get_interpreter()
        if interpreter is None:
            return None, None

        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        input_shape = tuple(1 if int(x) < 0 else int(x) for x in input_detail['shape'])
        inp = sample.astype(np.float32)
        if np.prod(input_shape) == inp.size:
            inp = inp.reshape(input_shape)
        elif np.prod(input_shape[1:]) == inp.size:
            inp = inp.reshape((1,) + input_shape[1:])
        else:
            print(f"[RA-OCCUPANCY] model input shape mismatch: {input_shape}, sample={inp.shape}")
            return None, None

        input_dtype = input_detail['dtype']
        if input_dtype != np.float32:
            scale, zero_point = input_detail.get('quantization', (0.0, 0))
            if scale:
                inp = inp / scale + zero_point
            inp = np.clip(np.round(inp), np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)

        interpreter.set_tensor(input_detail['index'], inp)
        interpreter.invoke()

        out = interpreter.get_tensor(output_detail['index'])
        if output_detail['dtype'] != np.float32:
            scale, zero_point = output_detail.get('quantization', (0.0, 0))
            if scale:
                out = (out.astype(np.float32) - zero_point) * scale
        out = np.asarray(out).reshape(-1)
        if out.size >= 2:
            label = int(np.argmax(out))
            score = float(out[1])
        elif out.size == 1:
            score = float(out[0])
            label = 1 if score >= 0.5 else 0
        else:
            return None, None
        return label, score

    def _update_ra_occ_oa_state(self, d):
        p = self.config.algo_params.get('RA-OCCUPANCY', {})
        h = d.get('h_raw')
        h_bg = d.get('h_bg')
        mean_th = float(p.get('oa_mean_threshold', 0.0))
        d['oa_mean_threshold'] = mean_th
        d['oa_label'] = 0
        d['oa_score'] = None
        d['oa_status'] = 'empty'

        if h is None or h_bg is None:
            self._ra_occ_h_bg_buffer.clear()
            self._ra_occ_h_buffer.clear()
            return d

        d['oa_status'] = 'out'
        self._ra_occ_h_bg_buffer.append(h_bg)
        h_norm = self._ra_occ_normalize_h(h_bg)
        self._ra_occ_h_buffer.append(h_norm)
        self._remember_ra_occ_raw_window()
        if len(self._ra_occ_h_buffer) < self._ra_occ_h_buffer.maxlen:
            return d

        bg_stacked = np.stack(list(self._ra_occ_h_bg_buffer), axis=0)
        stacked = np.stack(list(self._ra_occ_h_buffer), axis=0)
        sample = np.transpose(stacked, (2, 1, 0))
        # sample = self._ra_occ_normalize_h(sample)
        # 能量阈值计算
        h_max = float(np.max(h))
        d['h_max'] = h_max
        if h_max < mean_th:

            return d

        self._export_ra_occ_playback_sample(sample)

        if self._ra_occ_model_enabled():
            label, score = self._ra_occ_run_model(sample)
        else:
            label, score = 1, 1.0
        if label is not None:
            d['oa_label'] = label
            d['oa_score'] = score
            d['oa_status'] = 'in' if label == 1 else 'out'
            if self.config.connection_mode in ('BD_UDP', 'BD_CAN'):
                if label == 1:
                    self._count_bd_trigger(score)
        return d

    def _ra_occ_stride_raw(self):
        p = self.config.algo_params.get('RA-OCCUPANCY', {})
        stride_combined = max(1, int(p.get('heatmap_update_stride_combined', 4)))
        cir_comb = max(1, int(p.get('cir_combine_num', 1)))
        return stride_combined * cir_comb

    def _ra_heatmap_ready(self):
        """RA-HEATMAP 是否可出图：整缓冲已满，或已攒够 heatmap_start_snapshots。"""
        if self.data_manager.buffer_full:
            return True
        p = self.config.algo_params.get('RA-CFAR', {})
        n_win = int(p.get('snapshots', 64))
        start = max(4, int(p.get('heatmap_start_snapshots', min(n_win, 16))))
        return self.data_manager.snapshot_count() >= start

    def _try_step_ra_occupancy(self):
        # [加速] 不必等整缓冲(540)填满: 攒够一个快照窗口(heatmap_start_snapshots)即开跑
        p0 = self.config.algo_params.get('RA-OCCUPANCY', {})
        start_min = max(4, int(p0.get('heatmap_start_snapshots', p0.get('snapshots', 64))))
        if not self.data_manager.ready_for(start_min):
            return None

        current_snapshot = self._ra_occ_snapshot_counter
        if (
            self._last_ra_occ_heatmap_snapshot is not None and
            current_snapshot - self._last_ra_occ_heatmap_snapshot < self._ra_occ_stride_raw()
        ):
            return None

        d = self.algo_processor.step_ra_occupancy()
        if d:
            self._last_ra_occ_heatmap_snapshot = current_snapshot
            _, _, state = self.ra_occupancy_detector.process(d['energy'])
            d['occupancy'] = state
            d['state'] = state
            self._update_ra_occ_oa_state(d)
        return d

    def loop(self):
        if not self.running: return
        
        # 1. 获取新帧
        new_frames = self.source.get_batch_frames()
        
        # 2. 【核心修复】只有当收到新数据时，才进行处理和导出
        if new_frames:
            # 获取当前配置中的第一个天线对，作为一轮的起点标记
            first_pair = (self.config.udp_tx_list[0], self.config.udp_rx_list[0])
            last_pair = (self.config.udp_tx_list[-1], self.config.udp_rx_list[-1])
            m = self.config.current_algo
            ra_occ_data = None

            for f in new_frames:
                if len(f) >= 4:
                    tx, rx, cir_data, raw_frame = f
                else:
                    tx, rx, cir_data = f
                    raw_frame = None
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
                    if (tx, rx) == first_pair:
                        self._ra_occ_snapshot_counter += 1
                    processed = self.data_manager.process_frame(tx, rx, cir_data)
                    if processed is not None:
                        cir_no_bg, _ = processed
                        self._bd_remember_bg_removed_frame(self._ra_occ_snapshot_counter, tx, rx, raw_frame, cir_no_bg)
                    if m == 'RA-OCCUPANCY' and (tx, rx) == last_pair:
                        # print(int(time.time()*1000))
                        d_ra_occ = self._try_step_ra_occupancy()
                        if d_ra_occ:
                            ra_occ_data = d_ra_occ
            
            # 只有在新帧触发了缓冲区更新后，才执行算法。
            # [加速] RA-HEATMAP 不必等整缓冲(540)填满: 攒够 heatmap_start_snapshots 即可开跑
            hm_early = (m == 'RA-HEATMAP' and self._ra_heatmap_ready())
            if self.data_manager.buffer_full or hm_early:
                d = None
                
                if m == 'PLOT':
                    d = self.algo_processor.step_waveform()
                elif m == '2D-MUSIC':
                    d = self.algo_processor.step_2d_music()
                elif m in ('POINT-CLOUD', 'POINT-CLOUD-OPTIMIZED', 'POINT-CLOUD-DUBHE', 'POINT-CLOUD-PAPER', 'RA-CFAR'):
                    if m == 'POINT-CLOUD':
                        d = self.algo_processor.step_point_cloud()
                    elif m == 'POINT-CLOUD-OPTIMIZED':
                        d = self.algo_processor.step_point_cloud_optimized()
                    elif m == 'POINT-CLOUD-PAPER':
                        d = self.algo_processor.step_point_cloud_paper()
                    elif m == 'RA-CFAR':
                        d = self.algo_processor.step_point_cloud_ra_cfar()
                    else:
                        d = self.algo_processor.step_point_cloud_dubhe()

                    # 导出逻辑 (三个点云模式共享)
                    if d and self.config.connection_mode == 'PLAYBACK':
                        d['filename'] = os.path.basename(self.config.playback_file)
                        if self.config.export_pc_json:
                            export_frame = {
                                "frame_index": len(self.pc_export_data),
                                "breath_val": d.get("breath_val", 0.0),
                                "points": d.get("detected_points", [])
                            }
                            self.pc_export_data.append(export_frame)

                    # 座椅占用检测
                    if d and self.config.algo_params.get('SEAT-OCCUPANCY', {}).get('enable', True):
                        occ, f_vals = self.seat_detector.process(
                            d.get('detected_points', []))
                        d['occupancy'] = occ
                        d['f_values'] = f_vals

                elif m == 'RA-HEATMAP':
                    # [加速] 只有出现新快照才重算热力图, 避免 50Hz tick 空转重算(A3)
                    if self._ra_occ_snapshot_counter != self._last_hm_run_snapshot:
                        d = self.algo_processor.step_ra_heatmap_view()
                        if d is not None:
                            self._last_hm_run_snapshot = self._ra_occ_snapshot_counter
                elif m == 'RA-OCCUPANCY':
                    d = ra_occ_data
                elif m == 'ANGLE-SPECTRUM':
                    d = self.algo_processor.step_angle_spectrum_view()
                elif m == 'AS-RAW':
                    d = self.algo_processor.step_angle_spectrum_raw()

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
    root = tk.Tk(); app = App(root); root.protocol("WM_DELETE_WINDOW", lambda: (save_config(app.config), app.stop(), root.destroy())); root.mainloop()
