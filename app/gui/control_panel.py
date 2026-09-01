"""控制面板：ControlPanel。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from app.config import ANTENNA_LAYOUTS
from app.gui.dialogs import AlgoSettingsDialog, SeatConfigDialog

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
        self.cbs['update_layout'](self.config.current_algo)

    def _open_algo(self):
        AlgoSettingsDialog(self, self.config.current_algo,
                           self.config.algo_params.get(self.config.current_algo, {}),
                           lambda p: (self.config.algo_params.update({self.config.current_algo: p}),
                                      save_config(self.config),
                                      self.cbs['update_layout'](self.config.current_algo)))

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
