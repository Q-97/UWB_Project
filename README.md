# UWB 车内占用检测（UWB Occupancy Detection）

基于 UWB/FMCW 雷达的上位机实时处理与占座检测项目：通过 CAN-FD / UDP / 串口接收下位机雷达 CIR 数据，做背景去除、RA 热力图、座位占用（empty/child/adult）判断，并提供 GUI 显示与录制回放。

## 目录结构

```
uwb-occupancy-detection/
├── gui_main.py              # 入口：运行 `python gui_main.py`（须在项目根目录启动）
├── config_save.json         # GUI 持久化配置（连接模式、算法参数、模型路径等）
├── app/                     # 运行时核心代码（GUI 依赖链）
│   ├── device_management.py # 设备抽象层（ZLG / TOOMOSS 二选一）
│   ├── CAN_data_listen.py   # CAN-FD 接收、UCI 包重组、标准帧封装
│   ├── UDP_data_listen.py   # UDP 接收、按 TX/RX 切分标准帧
│   ├── radar_config.py      # UCI 协议常量（命令 GID/OID）
│   ├── util.py              # 算法辅助工具
│   ├── breathe.py           # 呼吸特征提取
│   └── comm/                # CAN 硬件驱动层
│       ├── zlgcan.py        # ZLG USBCANFD 驱动封装
│       ├── canfd.py         # TOOMOSS USB2CANFD 驱动封装
│       ├── usb_device.py    # TOOMOSS USB2XXX 设备枚举
│       └── usb2canfd.py     # TOOMOSS CANFD 操作函数
├── experiments/             # 离线训练与可视化实验（独立于 GUI，PyTorch）
│   ├── torch_MxTx2.py       # M点×T帧×2坐标 → MLP（INT/EXT 二分类）
│   ├── torch_MxTx2_GRU.py   # 同上，GRU 结构
│   ├── torch_MxTx2_TMLP.py  # 同上，TemporalMLP / UnifiedRadarModel
│   ├── torch_Nx2.py         # N点×2坐标 → MLP
│   └── MxTx2_*_vis.py / Nx2_*_vis.py   # 样本/错误可视化
├── tools/                   # 工具脚本
│   ├── generator.py         # 点云 → RA 热力图矩阵/图片生成
│   ├── generator_bd.py      # 婴儿检测(BD)数据生成（依赖 generator）
│   └── rename_matrix_files.py  # 批量重命名矩阵文件
├── docs/                    # 文档
│   ├── RA_OCCUPANCY_UDP_FLOW.md
│   ├── REFACTOR_PLAN.md     # gui_main.py 模块化拆分规划
│   ├── Dubhe_CPD_Algo_Guide.md
│   ├── *.pdf / *.doc / *.docx
│   └── references/          # 原 pdf_output/（论文提取文本）
├── model/                   # TFLite 模型（OA/BD），路径由配置相对引用
├── data/                    # 采集/导出数据
├── libs/                    # TOOMOSS USB2XXX 跨平台 SDK（勿移动，见下）
├── kerneldlls/              # ZLG CAN 驱动内核 DLL（勿移动，见下）
├── zlgcan.dll / zlgcan.lib  # ZLG SDK 主库（勿移动，见下）
```

## 运行

```bash
# 必须在项目根目录启动（驱动 DLL 按当前工作目录相对加载）
python gui_main.py
```

连接模式在 GUI 的「Communication」区选择：`UDP` / `BD_UDP` / `BD_CAN` / `SERIAL` / `CAN` / `PLAYBACK`。

## 数据接收链路

```
下位机 CAN-FD (ID=0x100, 64B)
  → app/comm（ZLG 轮询批量接收）
  → app/CAN_data_listen.py（字节流找同步头 0F 00 10 01 → 320B UCI 包 → 4 块快照重组）
  → 138B 标准帧: [FF 00 FF 00][TX][RX][CIR 32×复数 int16][F0 00 F0 00]
  → gui_main.py（parse_frame → 背景去除 → RA 热力图 → 座位三态）
```

详见 `docs/RA_OCCUPANCY_UDP_FLOW.md`。

## 注意事项（为什么这些目录/文件留在根目录）

- `zlgcan.dll` / `zlgcan.lib` / `kerneldlls/` / `libs/` 是厂商 SDK，代码按 **CWD 相对路径**加载
  （`app/comm/zlgcan.py` 的 `./zlgcan.dll`、`app/comm/usb_device.py` 的 `os.getcwd()/libs/...`），
  移动会导致驱动初始化失败，请保持它们在根目录，并从根目录启动程序。
- `experiments/` 与 `tools/` 是独立脚本，不参与 GUI 运行，可单独 `python <脚本>` 执行；
  `generator*.py` 的输入/输出路径是硬编码的（如 `data/data_ori`、`../model/h_bg/...`），按需自行调整。
- `model/` 路径由 `config_save.json` 与 GUI 默认值相对引用，勿移动。

## 依赖

见 `requirements.txt`。核心运行需要：numpy、scipy、matplotlib、pyserial；GUI 用 tkinter（Python 自带）。
OA/BD 模型推理需要 `tensorflow` 或 `tflite-runtime`（可选，模型缺失时优雅降级）；
`experiments/` 训练需要 `torch`（+ `pandas` 用于 TMLP 汇总）。
