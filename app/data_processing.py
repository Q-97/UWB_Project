"""数据层：FixedBuffer / BackgroundRemoval / RadarDataManager。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
from collections import deque
from typing import Optional, Tuple

import numpy as np

from app.config import RadarConfig

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
