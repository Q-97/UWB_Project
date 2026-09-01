"""绘图面板：PlotPanel（11 种显示模式）。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
import tkinter as tk

import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

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

