"""编排层：App（原 gui_main.py 的上帝类，阶段 1 整体搬迁，阶段 3.4 瘦身）。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.algorithms import AlgorithmProcessor
from app.config import NpEncoder, PROJECT_ROOT, RadarConfig, load_config, save_config
from app.data_processing import RadarDataManager
from app.detectors import RAOccupancyDetector, SeatOccupancyDetector
from app.gui.control_panel import ControlPanel
from app.gui.plot_panel import PlotPanel
from app.protocol import RadarProtocol
from app.ra_occ_model import OccupancyModelService
from app.sources import FilePlaybackSource, LiveRadarSource

class App:
    def __init__(self, root):
        self.root = root
        self.config = RadarConfig(); self.config.load_layout("2x4_Default")
        load_config(self.config)  # 持久化: 用已保存的配置覆盖默认值
        self.root.geometry(self.config.gui.get('window_size', '1300x850'))
        if not os.path.exists(self.config.data_save_dir): os.makedirs(self.config.data_save_dir)
        self.data_manager = RadarDataManager(self.config); self.algo_processor = AlgorithmProcessor(self.config, self.data_manager)
        self.source = None; self.running = False
        self.root.columnconfigure(1, weight=1); self.root.rowconfigure(0, weight=1)
        cbs = {'start': self.start, 'stop': self.stop, 'rec_start': self.rec_start, 'rec_stop': self.rec_stop, 'update_layout': self.update_layout}
        self.ctrl = ControlPanel(root, self.config, cbs); self.ctrl.grid(row=0, column=0, sticky='ns')
        self.plot_panel = PlotPanel(root); self.plot_panel.grid(row=0, column=1, sticky='nsew')
        self.plot_panel.set_theme(self.config.gui.get('colors', {}))
        self.plot_panel.init_layout("PLOT", self.config.algo_params['PLOT'])
        self.pc_export_data = []  # 新增：用于存储点云导出数据的列表
        self.playback_queue = [] # 新增：存放待处理的文件路径队列
        self.playback_skip_state = False  # False 表示处理，True 表示跳过
        # 座椅占用检测器
        occ_params = self.config.algo_params.get('SEAT-OCCUPANCY', {})
        self.seat_detector = SeatOccupancyDetector(occ_params)
        self.ra_occupancy_detector = RAOccupancyDetector(occ_params)
        # OA/BD 模型服务（阶段 3.4：原 App 内 _ra_occ_* / _bd_* 方法迁出）
        self.occ_service = OccupancyModelService(
            self.config, self.data_manager, self.algo_processor, self.ra_occupancy_detector)
    def update_layout(self, mode):
        # 1. 更新绘图面板的布局
        p = self.config.algo_params.get(mode, {})
        occ_cfg = self.config.algo_params.get('SEAT-OCCUPANCY', {})
        p['occupancy_config'] = occ_cfg
        self.plot_panel.init_layout(mode, p)

        # 2. 重建座椅检测器 (使配置修改立即生效)
        self.seat_detector = SeatOccupancyDetector(occ_cfg)
        self.ra_occupancy_detector = RAOccupancyDetector(occ_cfg)
        # 2.5 OA/BD 服务重建（绑定新检测器 + 清空缓冲）
        self.occ_service = OccupancyModelService(
            self.config, self.data_manager, self.algo_processor, self.ra_occupancy_detector)

        # 3. 【核心修复】强制算法处理器重新初始化参数
        # 这样当你修改了虚拟天线索引、频率范围等参数时，后端才会重新计算
        self.algo_processor.init_done = False
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
                self.occ_service.bd_trigger_count = 0
            self.occ_service.reset()
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
        self.plot_panel.ra_occ_oa_filter_history.clear()
        self.occ_service = OccupancyModelService(
            self.config, self.data_manager, self.algo_processor, self.ra_occupancy_detector)
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
                        self.occ_service._ra_occ_snapshot_counter += 1
                    processed = self.data_manager.process_frame(tx, rx, cir_data)
                    if processed is not None:
                        cir_no_bg, _ = processed
                        self.occ_service.remember_bg_removed_frame(
                            self.occ_service._ra_occ_snapshot_counter, tx, rx, raw_frame, cir_no_bg)
                    if m == 'RA-OCCUPANCY' and (tx, rx) == last_pair:
                        d_ra_occ = self.occ_service.try_step_ra_occupancy()
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
