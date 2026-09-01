"""回归冒烟脚本：验证解帧 / 配置 / 数据管理 / 算法流水线在重构后行为不回归。

用法（在项目根目录，py310 环境）:
    python tools/regression_smoke.py [--bin 路径]

说明：
- 主线程直读 .bin（不走回放线程限速，避免 Windows time.sleep 15.6ms 粒度拖慢测试）；
- 覆盖：协议解析、配置加载、ConfigAdapter、DataManager 背景去除、RA-OCCUPANCY /
  POINT-CLOUD-OPTIMIZED / RA-HEATMAP 流水线实跑、座椅状态输出。
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default="data/in_a1_d1_s1_1.bin")
    args = parser.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        print(f"[smoke] 找不到回放文件: {bin_path}")
        return 2

    import numpy as np

    from app.algorithms import AlgorithmProcessor
    from app.config import RadarConfig, load_config
    from app.data_processing import RadarDataManager
    from app.detectors import RAOccupancyDetector
    from app.protocol import RadarProtocol

    # 1) 配置加载
    cfg = RadarConfig()
    load_config(cfg)
    print(f"[smoke] config mode={cfg.connection_mode} algo={cfg.current_algo}")

    # 2) 协议解析 + 数据管理
    cfg.ft_len = 32
    cfg.max_snapshots = 144
    RadarProtocol.update_protocol(32)
    data = bin_path.read_bytes()
    n_frames = len(data) // RadarProtocol.FRAME_LEN
    dm = RadarDataManager(cfg)
    ap = AlgorithmProcessor(cfg, dm)
    bad = 0
    t0 = time.time()
    for i in range(n_frames):
        parsed = RadarProtocol.parse_frame(data[i * 138:(i + 1) * 138])
        if parsed is None or parsed[2].shape != (32,):
            bad += 1
            continue
        dm.process_frame(parsed[0], parsed[1], parsed[2])
    print(f"[smoke] parsed+processed {n_frames} frames, bad={bad}, {time.time() - t0:.2f}s")
    if bad or not dm.buffer_full:
        print("[smoke] FAIL: 坏帧或缓冲区未满")
        return 1

    # 3) 算法流水线
    d_ra = ap.step_ra_occupancy()
    if d_ra is None or 'heatmap' not in d_ra:
        print("[smoke] FAIL: RA-OCCUPANCY")
        return 1
    occ = RAOccupancyDetector(cfg.algo_params.get('SEAT-OCCUPANCY', {}))
    _, _, state = occ.process(d_ra['energy'])
    print(f"[smoke] RA-OCCUPANCY heatmap={d_ra['heatmap'].shape} state={state}")

    d_pc = ap.step_point_cloud_optimized()
    if d_pc is None:
        print("[smoke] FAIL: POINT-CLOUD-OPTIMIZED")
        return 1
    print(f"[smoke] POINT-CLOUD-OPTIMIZED points={len(d_pc.get('points', []))}")

    if ap.step_ra_heatmap_view() is None:
        print("[smoke] FAIL: RA-HEATMAP")
        return 1
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
