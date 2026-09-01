"""对话框：AlgoSettingsDialog / SeatConfigDialog。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
import json

import tkinter as tk
from tkinter import ttk

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
