"""编排层：App（原 gui_main.py 的上帝类，阶段 1 整体搬迁）。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
import json
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.algorithms import AlgorithmProcessor
from app.config import NpEncoder, PROJECT_ROOT, RadarConfig, load_config, save_config
from app.data_processing import RadarDataManager
from app.detectors import RAOccupancyDetector, SeatOccupancyDetector
from app.gui.control_panel import ControlPanel
from app.gui.plot_panel import PlotPanel
from app.protocol import RadarProtocol
from app.sources import FilePlaybackSource, LiveRadarSource

class App:
    def __init__(self, root):
        self.root = root; self.root.geometry("1300x850"); self.config = RadarConfig(); self.config.load_layout("2x4_Default")
        load_config(self.config)  # 持久化: 用已保存的配置覆盖默认值
        if not os.path.exists(self.config.data_save_dir): os.makedirs(self.config.data_save_dir)
        self.data_manager = RadarDataManager(self.config); self.algo_processor = AlgorithmProcessor(self.config, self.data_manager)
        self.source = None; self.running = False
        self.root.columnconfigure(1, weight=1); self.root.rowconfigure(0, weight=1)
        cbs = {'start': self.start, 'stop': self.stop, 'rec_start': self.rec_start, 'rec_stop': self.rec_stop, 'update_layout': self.update_layout}
        self.ctrl = ControlPanel(root, self.config, cbs); self.ctrl.grid(row=0, column=0, sticky='ns')
        self.plot_panel = PlotPanel(root); self.plot_panel.grid(row=0, column=1, sticky='nsew')
        self.plot_panel.init_layout("PLOT", self.config.algo_params['PLOT'])
        self._ra_occ_snapshot_counter = 0
        self._last_ra_occ_heatmap_snapshot = None
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
        output_dir = os.path.join(PROJECT_ROOT, playback_name)
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
        base_dir = PROJECT_ROOT
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

    def _try_step_ra_occupancy(self):
        if not self.data_manager.buffer_full:
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
            
            # 只有在新帧触发了缓冲区更新后，才执行算法
            if self.data_manager.buffer_full:
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
                    d = self.algo_processor.step_ra_heatmap_view()
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

