"""配置层：NpEncoder / 项目根定位 / 配置持久化 / RadarConfig / 天线布局 / 算法默认参数。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
import copy
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 项目根目录（本文件位于 app/ 下，上溯一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config_save.json')

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

# ---------------------------------------------------------------------------
# 配置分区（阶段 2：硬编码 → 可配置）
# 原则：默认值 = 现状，未配置时行为不变
# ---------------------------------------------------------------------------
DEFAULT_SECTIONS = {
    "paths": {
        "model_dir": "./model",            # 模型根目录（OA/BD .tflite 相对此解析）
        "oa_model": "",                     # 默认 OA 模型路径（空 = 沿用 algo_params 里的 oa_model_path）
        "data_dir": "./data",
        "playback_export_dir": "",          # 回放样本导出目录（空 = 项目根 / <回放文件名>）
    },
    "gui": {
        "title": "Radar V22 (Fixed)",
        "window_size": "1300x850",
        "speech_enabled": True,
        "speech_texts": {"start": "开始采样", "stop": "结束采样"},
        "colors": {
            "state": {"empty": "green", "child": "gold", "adult": "red"},
            "seat": {"empty": "green", "child": "gold", "adult": "red"},
            "panel_bg": "#f0f0f0",
            "heatmap_bg": "#f2f2f2",
        },
    },
    "protocol": {
        "ft_len": 32,
        "start_sign": "ff00ff00",           # 标准帧起始魔数（hex 字符串）
        "stop_sign": "f000f000",            # 标准帧结束魔数
    },
    "recording": {
        # 录制文件名模板（{...} 占位符，默认值 = 原硬编码拼接逻辑）
        "in_template": "in_{person}_{area}{position}_{pose}_{seq}.bin",
        "out_template": "out_{person}_{seq}.bin",
        "default_person_id": "a1",
        "default_action_time": 10,
        "rename_suffix": "_{n}",
    },
}


def _merge_section(saved, defaults):
    """一级浅合并：saved 覆盖 defaults，dict 值做二级合并（保留 defaults 新增 key）。"""
    out = dict(defaults)
    if isinstance(saved, dict):
        for k, v in saved.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**out[k], **v}
            else:
                out[k] = v
    return out


def save_config(config):
    """将当前配置持久化到 JSON 文件（v2 分区格式）。"""
    try:
        data = {
            "version": 2,
            "paths": config.paths,
            "gui": config.gui,
            "protocol": config.protocol,
            "recording": config.recording,
            "radar_config": {k: getattr(config, k) for k in _SAVE_FIELDS if hasattr(config, k)},
            "algo_params": config.algo_params,
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=NpEncoder, ensure_ascii=False)
    except Exception as e:
        print(f"[Config] 保存配置失败: {e}")


def load_config(config):
    """从 JSON 文件加载配置（兼容 v1/v2 格式）, 覆盖默认值. 返回 True 表示加载成功."""
    if not os.path.exists(CONFIG_FILE):
        return False
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 分区配置: 合并（v1 文件无分区 → 使用默认值）
        for section in DEFAULT_SECTIONS:
            setattr(config, section, _merge_section(data.get(section), DEFAULT_SECTIONS[section]))
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
        validate_algo_params(config.algo_params)
        # 协议配置（帧魔数/帧长可配置）
        configure_protocol(config.protocol, ft_len=getattr(config, 'ft_len', 32))
        # 确保天线布局与 current_layout_name 一致
        if 'RA-OCCUPANCY' in config.algo_params and 'range_bin_keep_range' in config.algo_params['RA-OCCUPANCY']:
            config.algo_params['RA-OCCUPANCY'].pop('range_bin_drop_front', None)
        config.load_layout(config.current_layout_name)
        print(f"[Config] 配置已从 {CONFIG_FILE} 加载 (v{data.get('version', 1)})")
        return True
    except Exception as e:
        print(f"[Config] 加载配置失败: {e}")
        return False


def configure_protocol(protocol_cfg, ft_len=None):
    """把 protocol 分区应用到 RadarProtocol（帧魔数/帧长可配置）。

    默认值 = 现状：start=ff00ff00 stop=f000f000 ft_len=32。
    """
    from app.protocol import RadarProtocol  # 局部导入，避免循环依赖
    start = bytes.fromhex(str(protocol_cfg.get('start_sign', 'ff00ff00')))
    stop = bytes.fromhex(str(protocol_cfg.get('stop_sign', 'f000f000')))
    f = ft_len if ft_len is not None else int(protocol_cfg.get('ft_len', 32))
    RadarProtocol.update_protocol(f, start_sign=start, stop_sign=stop)


def get_algo(config, name):
    """算法参数的深拷贝视图（防止外部修改污染默认值）。"""
    return copy.deepcopy(config.algo_params.get(name, {}))


def validate_algo_params(algo_params):
    """轻量校验已知算法参数的类型/数值范围，非法值回落默认并告警。"""
    rules = {
        "RA-OCCUPANCY": {
            "heatmap_update_stride_combined": (int, 1, None),
            "azimuth_num": (int, 1, None),
            "snapshots": (int, 1, None),
        },
        "POINT-CLOUD-OPTIMIZED": {"snapshots": (int, 1, None)},
        "POINT-CLOUD-DUBHE": {"ring_buffer_len": (int, 1, None), "slide_step": (int, 1, None)},
    }
    for algo, checks in rules.items():
        p = algo_params.get(algo)
        if not isinstance(p, dict):
            continue
        for key, (typ, lo, hi) in checks.items():
            if key not in p:
                continue
            v = p[key]
            try:
                ok = isinstance(v, typ) and (lo is None or v >= lo) and (hi is None or v <= hi)
            except TypeError:
                ok = False
            if not ok:
                default = DEFAULT_ALGO_PARAMS.get(algo, {}).get(key)
                print(f"[Config] 警告: {algo}.{key}={v!r} 非法，回落默认 {default!r}")
                p[key] = default

# ==============================================================================
# 1. Presets & Defaults
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
    algo_params: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_ALGO_PARAMS))

    # --- 配置分区（阶段 2：硬编码 → 可配置；默认值 = 现状）---
    paths: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SECTIONS["paths"]))
    gui: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SECTIONS["gui"]))
    protocol: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SECTIONS["protocol"]))
    recording: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SECTIONS["recording"]))
    current_algo: str = "PLOT"
    
    # --- 杂项 ---
    snapshot_rate: int = 20
    tap_interval_si: float = 1e-9 # 硬件采样间隔，通常固定
    data_save_dir: str = "./data"
    record_filename: str = "radar_capture.bin"  # <--- [新增] 默认文件名
    record_duration: float = 10.0
    playback_file_list: List[str] = field(default_factory=list)  # 修复：原类级共享可变默认值
    playback_file: str = ""
    playback_duration: float = 15.0  # [新增] 期望的回放总时长（秒）
    export_pc_json: bool = False  # 新增：是否在回放时导出点云 JSON
    recent_save_dirs: List[str] = field(default_factory=lambda: ["./data"])
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
