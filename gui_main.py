"""兼容入口（薄 shim）。

原 5769 行实现已按 docs/GUI_REFACTOR_PLAN.md 阶段 1 拆分到 app/ 包：

- app/config.py           配置层（NpEncoder / 配置持久化 / RadarConfig / 布局 / 算法默认参数）
- app/protocol.py         协议层（RadarProtocol / ConfigAdapter）
- app/data_processing.py  数据层（FixedBuffer / BackgroundRemoval / RadarDataManager）
- app/algorithms.py       算法层（AlgorithmProcessor 9 条流水线）
- app/detectors.py        检测器（SeatOccupancyDetector / RAOccupancyDetector）
- app/sources.py          数据源（LiveRadarSource / FilePlaybackSource）
- app/gui/                界面层（plot_panel / control_panel / dialogs）
- app/gui_app.py          编排层（App）
- app/main.py             入口

保留本文件仅为兼容旧启动方式（PyCharm 运行配置 / 命令行 python gui_main.py）。
"""
from app.main import main

if __name__ == '__main__':
    main()
