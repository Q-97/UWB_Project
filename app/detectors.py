"""检测器：SeatOccupancyDetector / RAOccupancyDetector。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

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
