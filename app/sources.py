"""数据源：LiveRadarSource / FilePlaybackSource。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
import json
import os
import queue
import threading
import time
from dataclasses import asdict
from datetime import datetime

import serial

from app.CAN_data_listen import CANFrameServer, radar_hardware_init
from app.UDP_data_listen import UDPFrameServer
from app.config import RadarConfig
from app.device_management import deinit_device, init_device
from app.protocol import ConfigAdapter, RadarProtocol

# ==============================================================================
# 5. IO Layer
# ==============================================================================
class LiveRadarSource:
    def __init__(self, config: RadarConfig):
        self.config = config; self.running = False; self.thread = None
        self.data_queue = queue.Queue(maxsize=5000); self.udp_server = None; self.can_server = None
        self.is_recording = False; self.record_file = None; self.record_lock = threading.Lock()
        self.rec_start_time = 0; self.rec_duration_target = 0; self.measured_fps = 0.0; self.last_fps_time = time.time(); self.frame_count_sec = 0
    def start(self):
        self.running = True; RadarProtocol.update_protocol(self.config.ft_len)
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
            from app.device_management import deinit_device
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
                elif self.config.connection_mode == 'SERIAL': raw_chunk = ser.read(ser.in_waiting) if ser.in_waiting else b''; time.sleep(0.002) if not raw_chunk else None
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
