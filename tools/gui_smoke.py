"""GUI 冒烟：真实构造 App + 切换算法布局（窗口隐藏），捕获构造函数/布局重建期运行时错误。

用法（项目根目录，py310）:
    python tools/gui_smoke.py
"""
import sys
import tkinter as tk
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_MODES = ['PLOT', '2D-MUSIC', 'POINT-CLOUD', 'POINT-CLOUD-OPTIMIZED',
          'POINT-CLOUD-DUBHE', 'POINT-CLOUD-PAPER', 'RA-CFAR',
          'RA-HEATMAP', 'RA-OCCUPANCY', 'ANGLE-SPECTRUM', 'AS-RAW']


def main() -> int:
    from app.gui_app import App

    root = tk.Tk()
    root.withdraw()  # 隐藏窗口，避免闪烁
    try:
        app = App(root)
        print(f"[gui_smoke] App constructed OK (mode={app.config.connection_mode})")
        for mode in _MODES:
            if mode in app.config.algo_params:
                app.config.current_algo = mode
                app.update_layout(mode)
        print(f"[gui_smoke] update_layout x{len(_MODES)} OK")
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
