"""OA/BD 模型服务：RA-OCCUPANCY 在线推理与 BD(婴儿检测) 采样导出。

阶段 3.4 从 App 抽出：依赖注入 config / data_manager / algo_processor / ra_occupancy_detector，
无 tkinter/GUI 依赖，可独立单测。
"""
import json
import os
from collections import deque
from dataclasses import asdict
from datetime import datetime

import numpy as np

from app.config import NpEncoder, PROJECT_ROOT
from app.protocol import RadarProtocol


class OccupancyModelService:
    """RA-OCCUPANCY OA 模型推理与 BD 采样导出服务。"""

    def __init__(self, config, data_manager, algo_processor, ra_occupancy_detector):
        self.config = config
        self.data_manager = data_manager
        self.algo_processor = algo_processor
        self.ra_occupancy_detector = ra_occupancy_detector
        self.reset()

    def reset(self):
        """清空全部状态缓冲与计数器（等价于原 App 的逐项清空）。"""
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

    # ------------------------------------------------------------------
    # 以下方法原样搬迁自 App（原 gui_main.py 5186+），行为不变
    # ------------------------------------------------------------------
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

    def remember_bg_removed_frame(self, snapshot_idx, tx, rx, raw_frame, cir_no_bg):
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
        export_root = self.config.paths.get('playback_export_dir') or PROJECT_ROOT
        output_dir = os.path.join(export_root, playback_name)
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
            model_path = self.config.paths.get('oa_model') or "./model/epoch-25-val-f1-100.0-sp-100.0.tflite"
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

    def try_step_ra_occupancy(self):
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