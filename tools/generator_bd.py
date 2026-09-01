from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# generator 与 generator_bd 同在 tools/ 下：兼容两种运行方式
# （`python tools/generator_bd.py` 直接运行 / 从项目根目录 import）
try:
    from tools import generator as gen
except ImportError:
    import generator as gen


BD_PROCESS_ENABLE = True
BD_TFLITE_MODEL_PATH = "../model/h_bg/epoch-37-val-f1-100.0-recall-100.0.tflite"
BD_POSITIVE_LABEL = 1
H_MAX_THRESHOLD = 0.045


def is_bd_process_enabled() -> bool:
    env_value = os.environ.get("BD_PROCESS_ENABLE")
    if env_value is None:
        return bool(BD_PROCESS_ENABLE)
    return env_value.strip().lower() not in {"0", "false", "no", "off"}


def resolve_model_path(model_path: str, base_dir: Path) -> Path:
    model_path = os.environ.get("BD_TFLITE_MODEL_PATH", model_path)
    if not model_path:
        raise ValueError("BD_TFLITE_MODEL_PATH is empty")
    path = Path(model_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_tflite_interpreter(model_path: Path):
    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        from tflite_runtime.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def reshape_model_input(sample: np.ndarray, input_shape: tuple[int, ...]) -> Optional[np.ndarray]:
    shape = tuple(1 if int(x) < 0 else int(x) for x in input_shape)
    if np.prod(shape) == sample.size:
        return sample.reshape(shape)
    if len(shape) > 1 and np.prod(shape[1:]) == sample.size:
        return sample.reshape((1,) + shape[1:])
    return None


def quantize_for_model(inp: np.ndarray, input_detail: dict) -> np.ndarray:
    input_dtype = input_detail["dtype"]
    if input_dtype == np.float32:
        return inp.astype(np.float32)

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if scale:
        inp = inp / scale + zero_point

    if np.issubdtype(input_dtype, np.integer):
        info = np.iinfo(input_dtype)
        inp = np.clip(np.round(inp), info.min, info.max)
    return inp.astype(input_dtype)


def dequantize_from_model(out: np.ndarray, output_detail: dict) -> np.ndarray:
    if output_detail["dtype"] == np.float32:
        return out

    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if scale:
        return (out.astype(np.float32) - zero_point) * scale
    return out.astype(np.float32)


def run_tflite_model(interpreter, sample: np.ndarray):
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    inp = sample.astype(np.float32)
    inp = reshape_model_input(inp, tuple(input_detail["shape"]))
    if inp is None:
        print(
            f"model input shape mismatch: expected={tuple(input_detail['shape'])}, sample={sample.shape}"
        )
        return None, None

    interpreter.set_tensor(input_detail["index"], quantize_for_model(inp, input_detail))
    interpreter.invoke()

    out = interpreter.get_tensor(output_detail["index"])
    out = dequantize_from_model(np.asarray(out), output_detail).reshape(-1)
    if out.size >= 2:
        label = int(np.argmax(out))
        score = (
            float(out[BD_POSITIVE_LABEL])
            if BD_POSITIVE_LABEL < out.size
            else float(out[label])
        )
    elif out.size == 1:
        score = float(out[0])
        label = 1 if score >= 0.5 else 0
    else:
        return None, None
    return label, score


def process_bd_bin_file(
    bin_path: Path,
    bd_file_index: int,
    matrix_output_dir: Path,
    image_output_dir: Path,
    radar_cfg: dict,
    params: dict,
    bd_interpreter,
    fig_save_flag: bool = True,
):
    protocol = gen.RadarProtocol(radar_cfg["ft_len"])
    data_manager = gen.RadarDataManager(radar_cfg)
    capon_angles, capon_sv = gen.init_capon_steering(params)

    first_pair = (radar_cfg["udp_tx_list"][0], radar_cfg["udp_rx_list"][0])
    last_pair = (radar_cfg["udp_tx_list"][-1], radar_cfg["udp_rx_list"][-1])
    cir_comb = max(1, int(params.get("cir_combine_num", 1)))
    stride_raw = max(1, int(params.get("heatmap_update_stride_combined", 8))) * cir_comb

    snapshot_counter = 0
    last_heatmap_snapshot: Optional[int] = None
    output_index = 0
    heatmap_index = 0
    matrix_output_index = 0
    positive_count = 0
    bd_sample_index = 0
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

            result = gen.make_heatmap_outputs(all_c, params, capon_angles, capon_sv)
            if result is None:
                continue

            h, h_bg, h_cart, _, _ = result
            output_index += 1
            last_heatmap_snapshot = snapshot_counter
            if output_index == 1:
                continue

            heatmap_index += 1
            image_name = f"{bin_path.stem}_{heatmap_index}"
            if fig_save_flag:
                gen.save_heatmap_image(h_cart, image_output_dir / f"{image_name}.png", params)
            h_norm = gen.normalize_data(h_bg)
            matrix_group.append(h_norm)
            if len(matrix_group) == gen.MATRIX_PATH_COUNT:
                stacked = np.stack(matrix_group, axis=0)
                h_max = float(np.max(stacked))
                h_max_threshold = float(params.get("oa_mean_threshold", H_MAX_THRESHOLD))
                if h_max < h_max_threshold:
                    print(
                        f"{bin_path.name}: candidate skipped by H_MAX "
                        f"({h_max:.6f} < {h_max_threshold:.6f})"
                    )
                    matrix_group.pop(0)
                    continue

                output_matrix = gen.normalize_data(np.transpose(stacked, (2, 1, 0)))
                matrix_output_index += 1
                label, score = run_tflite_model(bd_interpreter, output_matrix)
                if label == BD_POSITIVE_LABEL:
                    positive_count += 1
                    bd_sample_index += 1
                    matrix_name = f"bd_{bd_file_index}_{bd_sample_index}"
                    gen.save_matrix_sample(output_matrix, matrix_output_dir / f"{matrix_name}.txt")

                score_text = "None" if score is None else f"{score:.6f}"
                print(
                    f"{bin_path.name}: candidate #{matrix_output_index}, label={label}, score={score_text}"
                )
                matrix_group.pop(0)

    print(
        f"{bin_path.name}: saved {positive_count}/{matrix_output_index} positive BD matrix sample(s), {heatmap_index} heatmap(s)"
        + (f", discarded {len(matrix_group)} ungrouped matrix/matrices" if matrix_group else "")
        + (f", skipped {skipped_frames} malformed frame(s)" if skipped_frames else "")
    )
    return positive_count


def main():
    base_dir = gen._project_root()
    if not is_bd_process_enabled():
        print("BD processing disabled; no BD samples will be processed.")
        return

    data_dir = base_dir.parent / gen.DATA_DIR_NAME
    output_root_dir = base_dir.parent / gen.OUTPUT_ROOT_DIR_NAME
    matrix_output_dir = output_root_dir / gen.MATRIX_OUTPUT_DIR_NAME
    image_output_dir = output_root_dir / gen.IMAGE_OUTPUT_DIR_NAME
    matrix_output_dir.mkdir(parents=True, exist_ok=True)
    if gen.fig_save_flag:
        image_output_dir.mkdir(parents=True, exist_ok=True)

    radar_cfg, params = gen.load_project_config(base_dir)
    params.setdefault("oa_mean_threshold", H_MAX_THRESHOLD)
    bin_files = sorted(
        bin_path for bin_path in data_dir.glob("*.bin")
        if "bd" in bin_path.stem.lower()
    )
    if not bin_files:
        print(f"No BD .bin files found in {data_dir}")
        return

    model_path = resolve_model_path(BD_TFLITE_MODEL_PATH, base_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"BD TFLite model not found: {model_path}")
    bd_interpreter = load_tflite_interpreter(model_path)
    bd_positive_total = 0

    print(f"BD input folder: {data_dir}")
    print(f"BD matrix output folder: {matrix_output_dir}")
    print(f"BD image output folder: {image_output_dir if gen.fig_save_flag else 'disabled'}")
    print(f"BD TFLite model: {model_path}")
    print(f"range_bin_keep_range={gen.RANGE_BIN_KEEP_RANGE}")
    print(f"heatmap_stride_combined={gen.HEATMAP_STRIDE_COMBINED}")

    for bd_file_index, bin_path in enumerate(bin_files, start=1):
        bd_positive_total += process_bd_bin_file(
            bin_path,
            bd_file_index,
            matrix_output_dir,
            image_output_dir,
            radar_cfg,
            params,
            bd_interpreter,
            gen.fig_save_flag,
        )
    print(f"BD positive samples saved: {bd_positive_total}")


if __name__ == "__main__":
    main()
