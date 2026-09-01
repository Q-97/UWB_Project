"""程序入口（薄）：原 gui_main.py 的 __main__ 块迁移至此。"""
import tkinter as tk

from app.config import save_config
from app.gui_app import App


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (save_config(app.config), app.stop(), root.destroy()))
    root.mainloop()


if __name__ == '__main__':
    main()