from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Optional

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_env] = "1"
os.environ["MKL_CBWR"] = "COMPATIBLE"

import matplotlib
import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 参数修改
# =========================

# 数据位置
DATA_DIR_NAME = "data/data_ori_3"

# 保存图片标志位
fig_save_flag = False

bdflag = True

# 每次滑动bin的数量
HEATMAP_STRIDE_COMBINED = 4

# 保留range bin范围，包含右端点；例如(5, 14)会丢弃0-4和15
RANGE_BIN_KEEP_RANGE = (5, 12)

# 放大和heatmap倍数
IMAGE_SCALE = 10
MATRIX_PATH_COUNT = 4
AZIMUTH_NUM = 24

# 读取config_save并覆盖GUI参数
USE_CONFIG_SAVE = False

# 结果保存位置
OUTPUT_TAG = f"combined={HEATMAP_STRIDE_COMBINED}_range={RANGE_BIN_KEEP_RANGE[0]}-{RANGE_BIN_KEEP_RANGE[1]}_path={MATRIX_PATH_COUNT}_azimuth={AZIMUTH_NUM}"
OUTPUT_ROOT_DIR_NAME = "data/data_input_3/4,4/(55,55)"
MATRIX_OUTPUT_DIR_NAME = f"2026_7_7_{OUTPUT_TAG}_matrix_h4_bg_norm"
IMAGE_OUTPUT_DIR_NAME = f"2026_7_7_{OUTPUT_TAG}_heatmap"
 
# =========================
# GUI参数
# =========================

DEFAULT_RADAR_CONFIG = {
    "ft_len": 32,
    "max_snapshots": 144,
    "udp_tx_list": [1, 2],
    "udp_rx_list": [4, 5, 6, 7],
    "bg_m_factor": 4,
    "num_tx_antennas": 2,
    "num_rx_antennas": 4,
}

DEFAULT_RA_OCCUPANCY_PARAMS = {
    "center_freq": 7.9872e9,
    "snapshots": 144,
    "cir_combine_num": 9,
    "leakage_offset": 5,
    "range_bin_keep_range": list(RANGE_BIN_KEEP_RANGE),
    "doppler_window": "chebyshev",
    "doppler_win_atten": 60,
    "doppler_dc_remove": True,
    "indices_azimuth": [2, 3, 6, 7],
    "ant_dbf_select": [2, 3, 6, 7],
    "azi_angle_range": [-55, 55],
    "azimuth_num": AZIMUTH_NUM,
    "ant_calib_en": True,
    "ant_calib_phase": [
        -0.0,
        -0.49072375380773364,
        1.1062868947533422,
        1.5748289100316295,
        -0.5876511374160144,
        -1.9895222818981357,
        0.12190304520431955,
        0.5787126081848506,
    ],
    "capon_diag_load": 1e-3,
    "dist_per_tap": 0.15,
    "smooth_kernel": [2, 4],
    "heatmap_update_stride_combined": HEATMAP_STRIDE_COMBINED,
    "plot_xlim": 1.5,
    "plot_ylim_min": -2.5,
    "plot_ylim_max": -0.1,
    "heatmap_clim_mode": "auto",
    "heatmap_clim_vmin": 0,
    "heatmap_clim_vmax": 100,
}


class RadarProtocol:
    START_SIGN = b"\xff\x00\xff\x00"
    HEADER_LEN = 4
    FOOTER_LEN = 4
    ANTENNA_INFO_LEN = 2

    def __init__(self, ft_len: int):
        self.ft_len = int(ft_len)
        self.cir_data_len = self.ft_len * 4
        self.frame_len = self.HEADER_LEN + self.ANTENNA_INFO_LEN + self.cir_data_len + self.FOOTER_LEN

    def parse_frame(self, frame_bytes: bytes):
        if len(frame_bytes) != self.frame_len:
            return None
        if not frame_bytes.startswith(self.START_SIGN):
            return None
        tx = frame_bytes[self.HEADER_LEN]
        rx = frame_bytes[self.HEADER_LEN + 1]
        cir = frame_bytes[self.HEADER_LEN + 2 : -self.FOOTER_LEN]
        s16 = np.frombuffer(cir, dtype=np.int16)
        c_data = s16[0::2].astype(np.float32) + 1j * s16[1::2].astype(np.float32)
        return tx, rx, c_data


class FixedBuffer:
    def __init__(self, maxlen: int):
        self.buffer = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def append(self, item):
        self.buffer.append(item)

    def get_data(self):
        return np.array(self.buffer) if self.buffer else np.array([])

    def __len__(self):
        return len(self.buffer)

    def is_full(self):
        return len(self.buffer) == self._maxlen


class BackgroundRemoval:
    def __init__(self, m_factor: float):
        self.m_factor = m_factor
        self.cir_ref: Optional[np.ndarray] = None
        self.cir_abs_ref: Optional[np.ndarray] = None
        self.first = True

    def update(self, r: np.ndarray):
        if self.first:
            self.cir_ref = r.copy()
            self.cir_abs_ref = np.abs(r)
            self.first = False
        else:
            self.cir_ref = (1 - (1 / self.m_factor)) * self.cir_ref + r / self.m_factor
            self.cir_abs_ref = (1 - (1 / self.m_factor)) * self.cir_abs_ref + np.abs(r) / self.m_factor

    def remove_background(self, r: np.ndarray):
        if self.first:
            self.update(r)
            return r, np.abs(r)
        r_no_bg = r - self.cir_ref
        r_abs_no_bg = np.abs(r) - self.cir_abs_ref
        self.update(r)
        return r_no_bg, r_abs_no_bg


class RadarDataManager:
    def __init__(self, radar_cfg: dict, skip_background_removal: bool = False):
        self.cfg = radar_cfg
        self.skip_background_removal = skip_background_removal
        self.pairs = [(tx, rx) for tx in radar_cfg["udp_tx_list"] for rx in radar_cfg["udp_rx_list"]]
        self.snapshots_data = {
            pair: {"complex": FixedBuffer(radar_cfg["max_snapshots"])}
            for pair in self.pairs
        }
        self.bgs = {pair: BackgroundRemoval(radar_cfg["bg_m_factor"]) for pair in self.pairs}
        self.buffer_full = False

    def process_frame(self, tx: int, rx: int, raw: np.ndarray):
        pair = (tx, rx)
        if pair not in self.snapshots_data:
            return
        if self.skip_background_removal:
            self.snapshots_data[pair]["complex"].append(raw)
        else:
            r_no_bg, _ = self.bgs[pair].remove_background(raw)
            self.snapshots_data[pair]["complex"].append(r_no_bg)
        if self.snapshots_data[self.pairs[0]]["complex"].is_full():
            self.buffer_full = True

    def min_len(self) -> int:
        return min(len(self.snapshots_data[p]["complex"]) for p in self.pairs)

    def get_all_snapshot_as_array(self):
        if not self.buffer_full:
            return None
        target_len = min(self.cfg["max_snapshots"], self.min_len())
        if target_len <= 0:
            return None

        arr = np.zeros(
            (
                self.cfg["num_rx_antennas"],
                self.cfg["num_tx_antennas"],
                self.cfg["ft_len"],
                target_len,
            ),
            dtype=np.complex64,
        )
        tx_map = {v: i for i, v in enumerate(self.cfg["udp_tx_list"])}
        rx_map = {v: i for i, v in enumerate(self.cfg["udp_rx_list"])}
        for (tx, rx), buffs in self.snapshots_data.items():
            if tx in tx_map and rx in rx_map:
                data_slice = buffs["complex"].get_data()[:target_len].T
                arr[rx_map[rx], tx_map[tx], :, :] = data_slice
        return arr


def load_project_config(base_dir: Path):
    radar_cfg = dict(DEFAULT_RADAR_CONFIG)
    params = dict(DEFAULT_RA_OCCUPANCY_PARAMS)
    config_path = base_dir / "config_save.json"

    if USE_CONFIG_SAVE and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        radar_cfg.update(saved.get("radar_config", {}))
        params.update(saved.get("algo_params", {}).get("RA-OCCUPANCY", {}))

    params["range_bin_keep_range"] = list(RANGE_BIN_KEEP_RANGE)
    params.pop("range_bin_drop_front", None)
    params["heatmap_update_stride_combined"] = HEATMAP_STRIDE_COMBINED
    return radar_cfg, params


def apply_range_bin_selection(current_cube: np.ndarray, params: dict) -> np.ndarray:
    keep_range = params.get("range_bin_keep_range")
    if keep_range is not None:
        if isinstance(keep_range, str):
            keep_range = keep_range.strip().strip("()[]").replace("\uff0c", ",").split(",")
        if len(keep_range) != 2:
            raise ValueError("range_bin_keep_range must be [start, end]")
        start_bin = int(keep_range[0])
        end_bin = int(keep_range[1])
        max_bin = current_cube.shape[2] - 1
        start_bin = max(0, min(start_bin, max_bin))
        end_bin = max(start_bin, min(end_bin, max_bin))
        return current_cube[:, :, start_bin:end_bin + 1, :]

    drop_front = int(params.get("range_bin_drop_front", 0) or 0)
    if drop_front > 0:
        drop_front = min(drop_front, max(0, current_cube.shape[2] - 1))
        return current_cube[:, :, drop_front:, :]

    leakage_offset = int(params.get("leakage_offset", 0))
    return np.roll(current_cube, -leakage_offset, axis=2)


def init_capon_steering(params: dict):
    wavelength = 2.99792458e8 / params.get("center_freq", 7.9872e9)
    all_virt_x = np.array([-0.038, 0.0, -0.038, -0.019, 0.0, 0.038, 0.0, 0.019])
    channels = np.array(params.get("indices_azimuth", [2, 3, 6, 7]), dtype=int)
    ant_x = all_virt_x[channels] / wavelength

    azi_deg = np.linspace(
        params["azi_angle_range"][0],
        params["azi_angle_range"][1],
        int(params["azimuth_num"]),
    )
    capon_angles = np.deg2rad(azi_deg)
    capon_sv = np.exp(1j * 2 * np.pi * ant_x[:, np.newaxis] * np.sin(capon_angles[np.newaxis, :]))

    if params.get("ant_calib_en", False):
        calib_phase = np.array(params.get("ant_calib_phase", [0.0] * 8), dtype=np.float64)
        capon_sv = capon_sv * np.exp(1j * calib_phase[channels]).reshape(-1, 1)

    return capon_angles, capon_sv


def compute_ra_heatmap(all_c: np.ndarray, params: dict, capon_angles: np.ndarray, capon_sv: np.ndarray):
    n_snaps = int(params["snapshots"])
    cir_comb = max(1, int(params.get("cir_combine_num", 1)))
    current_cube = all_c[:, :, :, -n_snaps:]

    if cir_comb > 1:
        n_comb = current_cube.shape[3] // cir_comb
        if n_comb <= 0:
            return None
        trim = n_comb * cir_comb
        current_cube = (
            current_cube[:, :, :32, :trim]
            .reshape(4, 2, 32, n_comb, cir_comb)
            .mean(axis=4)
        )

    current_cube = apply_range_bin_selection(current_cube, params)

    cube_flat = current_cube.transpose(1, 0, 2, 3).reshape(
        8, current_cube.shape[2], current_cube.shape[3]
    )
    valid_indices = params.get("indices_azimuth", [2, 3, 6, 7])
    a_all = cube_flat[valid_indices, :, :]

    k_count = a_all.shape[1]
    angle_count = len(capon_angles)
    slow_count = a_all.shape[2]
    diag_load = params.get("capon_diag_load", 1e-3)
    h = np.zeros((k_count, angle_count), dtype=np.float64)

    for k_idx in range(k_count):
        a_k = a_all[:, k_idx, :]
        r_k = (a_k @ a_k.conj().T) / slow_count
        r_k += np.eye(a_k.shape[0]) * diag_load * np.abs(np.trace(r_k))
        try:
            r_inv = np.linalg.inv(r_k)
        except np.linalg.LinAlgError:
            continue
        denom = np.sum((capon_sv.conj() * (r_inv @ capon_sv)), axis=0)
        h[k_idx, :] = 1.0 / np.real(np.clip(denom, 1e-12, None))

    ranges_m = np.arange(k_count) * params["dist_per_tap"]
    angles_deg = np.rad2deg(capon_angles)
    return h, ranges_m, angles_deg


def ra_to_cartesian(h: np.ndarray, ranges_m: np.ndarray, angles_deg: np.ndarray, params: dict):
    xlim = params.get("plot_xlim", 1.5)
    y_min = params.get("plot_ylim_min", -2.5)
    y_max = params.get("plot_ylim_max", -0.1)
    res = 0.05
    interp = RegularGridInterpolator(
        (ranges_m, angles_deg),
        h,
        bounds_error=False,
        fill_value=0.0,
    )

    xs = np.arange(-xlim, xlim + res, res)
    ys = np.arange(y_min, y_max + res, res)
    x_grid, y_grid = np.meshgrid(xs, ys)

    r_grid = np.sqrt(x_grid**2 + y_grid**2)
    a_grid = np.rad2deg(np.arctan2(x_grid, -y_grid))
    pts = np.stack([r_grid.ravel(), a_grid.ravel()], axis=1)
    h_cart = interp(pts).reshape(len(ys), len(xs))
    return h_cart, xs, ys


def make_heatmap_outputs(all_c: np.ndarray, params: dict, capon_angles: np.ndarray, capon_sv: np.ndarray):
    computed = compute_ra_heatmap(all_c, params, capon_angles, capon_sv)
    if computed is None:
        return None

    h, ranges_m, angles_deg = computed
    h_sq_noise = np.sqrt(np.sum(np.square(h)) / (h.shape[0] * h.shape[1]))
    # H_sq_mean = np.sum(H) / (H.shape[0] * H.shape[1])
    h_bg = h - h_sq_noise

    kernel_size = params.get("smooth_kernel", [2, 4])
    smooth_kernel = np.ones((kernel_size[0], kernel_size[1])) / (kernel_size[0] * kernel_size[1])
    h_bg = ndimage.convolve(h_bg, smooth_kernel, mode="reflect")

    h_cart, xs, ys = ra_to_cartesian(h_bg, ranges_m, angles_deg, params)
    return h, h_bg, h_cart, xs, ys

def normalize_data(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(matrix)
    std = np.std(matrix)
    if std <= eps:
        return matrix
    return (matrix - mean) / std

def save_heatmap_image(matrix: np.ndarray, path: Path, params: dict):
    if params.get("heatmap_clim_mode", "auto") == "fixed":
        vmin = params.get("heatmap_clim_vmin", 0)
        vmax = params.get("heatmap_clim_vmax", 100)
    else:
        valid = matrix[np.isfinite(matrix)]
        if valid.size:
            vmin = max(-80, np.percentile(valid, 5))
            vmax = np.max(valid)
            if vmin >= vmax:
                vmin = vmax - 1.0
        else:
            vmin, vmax = 0.0, 1.0

    height, width = matrix.shape
    dpi = 100
    fig = plt.figure(figsize=(width * IMAGE_SCALE / dpi, height * IMAGE_SCALE / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        matrix,
        cmap="jet",
        origin="lower",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_axis_off()
    fig.savefig(path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def save_matrix_sample(matrix: np.ndarray, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for idx in range(matrix.shape[0]):
            np.savetxt(f, matrix[idx], fmt="%.10e")


def process_bin_file(
    bin_path: Path,
    matrix_output_dir: Path,
    image_output_dir: Path,
    radar_cfg: dict,
    params: dict,
    fig_save_flag: bool = True,
):
    protocol = RadarProtocol(radar_cfg["ft_len"])
    is_bd_file = bdflag and "bd" in bin_path.stem.lower()
    data_manager = RadarDataManager(radar_cfg, skip_background_removal=is_bd_file)
    capon_angles, capon_sv = init_capon_steering(params)

    first_pair = (radar_cfg["udp_tx_list"][0], radar_cfg["udp_rx_list"][0])
    last_pair = (radar_cfg["udp_tx_list"][-1], radar_cfg["udp_rx_list"][-1])
    cir_comb = max(1, int(params.get("cir_combine_num", 1)))
    stride_raw = max(1, int(params.get("heatmap_update_stride_combined", 8))) * cir_comb

    snapshot_counter = 0
    last_heatmap_snapshot: Optional[int] = None
    output_index = 0
    heatmap_index = 0
    matrix_output_index = 0
    matrix_group = []
    skipped_frames = 0

    with bin_path.open("rb") as f:
        while True:
            chunk = f.read(protocol.frame_len)
            if not chunk:
                break
            if len(chunk) < protocol.frame_len:
                skipped_frames += 1
                break

            parsed = protocol.parse_frame(chunk)
            if parsed is None:
                skipped_frames += 1
                continue

            tx, rx, raw = parsed
            if (tx, rx) == first_pair:
                snapshot_counter += 1

            data_manager.process_frame(tx, rx, raw)

            if (tx, rx) != last_pair or not data_manager.buffer_full:
                continue

            should_update = (
                last_heatmap_snapshot is None
                or snapshot_counter - last_heatmap_snapshot >= stride_raw
            )
            if not should_update:
                continue

            all_c = data_manager.get_all_snapshot_as_array()
            if all_c is None:
                continue

            result = make_heatmap_outputs(all_c, params, capon_angles, capon_sv)

            if result is None:
                continue

            h, h_bg, h_cart, _, _ = result
            h_norm = normalize_data(h_bg)
            output_index += 1
            last_heatmap_snapshot = snapshot_counter
            if output_index == 1 and not is_bd_file:
                continue

            heatmap_index += 1
            image_name = f"{bin_path.stem}_{heatmap_index}"
            if fig_save_flag:
                save_heatmap_image(
                    h_cart,
                    image_output_dir / f"{image_name}.png",
                    params,
                )

            # matrix_group.append(h_norm)
            matrix_group.append(h_bg)
            if len(matrix_group) == MATRIX_PATH_COUNT:
                stacked = np.stack(matrix_group, axis=0)
                # output_matrix = np.transpose(stacked, (2, 1, 0))
                output_matrix = normalize_data(np.transpose(stacked, (2, 1, 0)))
                matrix_output_index += 1
                matrix_name = f"{bin_path.stem}_path={MATRIX_PATH_COUNT}_{matrix_output_index}"
                save_matrix_sample(output_matrix, matrix_output_dir / f"{matrix_name}.txt")
                matrix_group.pop(0)

    output_label = "matrices/heatmaps" if fig_save_flag else "matrices"
    print(
        f"{bin_path.name}: saved {matrix_output_index} matrix sample(s), {heatmap_index} heatmap(s)"
        + (f", discarded {len(matrix_group)} ungrouped matrix/matrices" if matrix_group else "")
        + (f", skipped {skipped_frames} malformed frame(s)" if skipped_frames else "")
    )


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / DATA_DIR_NAME
    output_root_dir = base_dir.parent / OUTPUT_ROOT_DIR_NAME
    matrix_output_dir = output_root_dir / MATRIX_OUTPUT_DIR_NAME
    image_output_dir = output_root_dir / IMAGE_OUTPUT_DIR_NAME
    matrix_output_dir.mkdir(parents=True, exist_ok=True)
    if fig_save_flag:
        image_output_dir.mkdir(parents=True, exist_ok=True)

    radar_cfg, params = load_project_config(base_dir)
    bin_files = sorted(data_dir.glob("*.bin"))
    if not bin_files:
        print(f"No .bin files found in {data_dir}")
        return

    print(f"Input folder: {data_dir}")
    print(f"Matrix output folder: {matrix_output_dir}")
    print(f"Image output folder: {image_output_dir if fig_save_flag else 'disabled'}")
    print(f"range_bin_keep_range={RANGE_BIN_KEEP_RANGE}")
    print(f"heatmap_stride_combined={HEATMAP_STRIDE_COMBINED}")

    for bin_path in bin_files:
        process_bin_file(bin_path, matrix_output_dir, image_output_dir, radar_cfg, params, fig_save_flag)


if __name__ == "__main__":
    main()
