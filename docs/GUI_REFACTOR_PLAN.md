# gui_main.py 解耦 / 优化 / 配置化 分步改造方案

> 版本：V2（详版） · 基于对 `gui_main.py`（5769 行）逐段核对后的结构盘点编写
> 前置：项目已按 `app/` 包结构整理（2025 年整理），本方案把 gui_main 的内容**拆入 `app/` 包内各模块**
> 参考：旧版 `docs/REFACTOR_PLAN.md`（聚焦拆包），本方案在其基础上补充**配置化改造**与**代码优化**两条主线，并更新行号
> 验证环境：`D:\Env\anaconda3\envs\py310`（PyCharm 项目 SDK）

---

## 改造进度（实施中自动更新）

| 阶段 | 内容 | 状态 | 说明 |
|---|---|---|---|
| 0 | 基线 | ✅ 部分 | 用现有 `data/in_a1_d1_s1_1.bin` 作基准；`tools/regression_smoke.py` 已建为回归工具（录制新样本需硬件） |
| 1 | 纯搬迁拆包 | ✅ 完成 | commit `a855203`；gui_main.py 变薄 shim；修复 3 处 `__file__` 路径基准 |
| 2 | 硬编码配置化 | ✅ 完成 | commit `0266625`；config_save.json v2 分区（paths/gui/protocol/recording）+ deepcopy + get_algo + 校验；顺带修复 `algorithms.py` 缺 `import time` |
| 3 | 解耦与优化 | 🔶 第一批 | commit `5f0973d`：FrameParser 注册表(3.1) / BaseSource 抽象(3.3) / 死代码清理(3.5) / TTS 封装(3.10) |
| 4 | 回归 | ⏳ 待做 | 需用户真机/窗口点检 |

阶段 3 剩余项（按风险排序，待续）：
- **3.7 统一双 DBF 缓存**（`_dbf_sv` vs `_dbf_sv_cache`，两处引导矢量构造方向/校准不同，需先核对再合并）
- **3.4 App 瘦身**（`_bd_*`/`_ra_occ_*` 迁往 `ra_occ_model.py`）
- **3.2 输出 TypedDict 契约**（9 条流水线返回值类型化）
- **3.6 三段共享预处理消重**（回归风险最高，须回放对比）
- **3.8 性能优化 / 3.9 类型注解**（低风险，随改随验）

---

## 0. 目标与原则

### 0.1 三大目标

| 目标 | 含义 | 验收口径 |
|---|---|---|
| **解耦** | 5769 行上帝文件拆成职责单一模块；GUI/算法/数据源/协议互不穿透 | `import` 无环；每个模块可独立单测 |
| **优化** | 消除死代码、重复预处理、全局可变状态、浅拷贝污染；补类型注解 | 行为不变的前提下代码量下降、结构清晰 |
| **配置化** | 路径、协议魔数、GUI 外观、录制命名规则等硬编码 → `config_save.json` 分区配置 | 不改代码即可调窗大小/颜色/帧格式/命名规则 |

### 0.2 铁律（每步必须遵守）

1. **先搬后改，行为不变**：每一步只做"原样搬迁 + 修正 import"，逻辑重构放第二阶段；
2. **小步可回退**：每个步骤一个 commit，任何时刻可 `git checkout` 回到上一步；
3. **每步有验证**：`python -m py_compile` + `import` 测试 +（条件允许时）PLAYBACK 回放对比输出；
4. **路径基准**：模块一律基于 `__file__` 定位项目根，不依赖 CWD（驱动 DLL 加载除外，见 README）。

---

## 1. 现状诊断（已核对，行号以此为准）

### 1.1 gui_main.py 结构地图

| 行号 | 内容 | 拆往目标模块 | 备注 |
|---|---|---|---|
| 1–36 | import + matplotlib 全局副作用 | 分散 | `matplotlib.use('TkAgg')`、SimHei rcParams 导入即生效 |
| 41–52 | `NpEncoder` | `app/config.py` | JSON 序列化辅助 |
| 54–108 | `CONFIG_FILE`/`_SAVE_FIELDS`/`save_config`/`load_config` | `app/config.py` | 与 RadarConfig 鸭子类型耦合 |
| 111–133 | `apply_range_bin_selection` | `app/algorithms.py` 或 `app/utils.py` | 被 AlgorithmProcessor 使用 |
| 139–152 | `ANTENNA_LAYOUTS`（2 套硬编码布局） | `app/config.py` | 建议可扩展为配置项 |
| 154–529 | `DEFAULT_ALGO_PARAMS`（9 套算法默认参数） | `app/config.py` | 大量魔数，见 §2 硬编码清单 |
| 534–553 | **`RadarProtocol`**（解帧入口） | **`app/protocol.py`** | 魔数/帧长硬编码；`update_protocol` 是**全局可变类状态** |
| 555–571 | `ConfigAdapter` | `app/protocol.py` | 桥接 RadarConfig → 帧服务器字段名 |
| 572–639 | `RadarConfig`（dataclass，30+ 字段） | `app/config.py` | 默认值硬编码；L614 `playback_file_list=[]` 是**类级共享可变默认值（bug）** |
| 644–759 | `FixedBuffer`/`BackgroundRemoval`/`RadarDataManager` | `app/data_processing.py` | 数据层 |
| 761 | `calculate_mdl_asc_local` | `app/algorithms.py` | 算法内部函数 |
| 774–3091 | **`AlgorithmProcessor`（约 2320 行，40+ 方法，9 条算法流水线）** | `app/algorithms.py` | **整类搬迁，不可按方法切**（兄弟方法互相调用） |
| 3093–3330 | `SeatOccupancyDetector`/`RAOccupancyDetector` | `app/detectors.py` | 零 GUI 依赖，拆包零风险 |
| 3332–3481 | `LiveRadarSource`/`FilePlaybackSource` | `app/sources.py` | 依赖 RadarProtocol（一并迁出） |
| 3483–3551 | `AlgoSettingsDialog` | `app/gui/dialogs.py` | |
| 3552–4345 | `PlotPanel`（11 种 mode） | `app/gui/plot_panel.py` | 颜色/坐标范围硬编码 |
| 4347–4568 | `SeatConfigDialog` | `app/gui/dialogs.py` | |
| 4570–5184 | `ControlPanel`（约 615 行） | `app/gui/control_panel.py` | 录制命名业务规则硬编码；TTS 走 PowerShell |
| 5186–5767 | **`App`（编排层，约 580 行）** | `app/gui_app.py` | 窗口 1300x850 硬编码；OA/BD 模型逻辑 ~350 行可抽离 |
| 5768–5769 | `__main__` 入口 | `app/main.py` | 薄入口 |

### 1.2 关键耦合事实

1. **依赖方向整体无环**：`ConfigAdapter→RadarProtocol`、`RadarDataManager→{FixedBuffer,BackgroundRemoval,RadarConfig}`、`App→{全部}`，子对象之间从不直接互引，全部经 App 中转 —— 最易拆的形态；
2. **循环导入唯一雷区**：`RadarConfig`、`RadarDataManager`、`calculate_mdl_asc_local` 与 `AlgorithmProcessor` 同文件；若 `algorithms.py` 反向 `import gui_main` 即死循环 → 公共符号先下沉，类型注解用 `TYPE_CHECKING`；
3. **AlgorithmProcessor 必须整体搬迁**：中段方法调用 8 个段外兄弟方法，`_angle_cfar_detect`/`_find_angle_peaks` 被后段 `_dbf_estimate` 反向调用；
4. **App 的鸭子类型契约（属性名不能改）**：`source.running`、`source.data_queue.empty()`、`source.measured_fps/playback_fps`、`get_rec_status()`、`get_progress()`、`get_batch_frames()`；`cbs` 回调 key 全集 = `{start, stop, rec_start, rec_stop, update_layout}`；`p['occupancy_config']` 隐式键；
5. **GUI 与算法只在契约层耦合**：`PlotPanel.update_data(mode, d)` 按 mode 硬编码读 dict key；`App.loop` 按 `config.current_algo` 字符串分发。算法模块**没有任何 tkinter/matplotlib 调用**；
6. **3 处路径依赖**：`CONFIG_FILE` 基于 `__file__`（拆包后基准变，须统一为 `app/config.py` 内定位项目根）；`oa_model_path="./model/…"` 相对项目根解析；录制/回放导出目录相对 `__file__`；
7. **`DEFAULT_ALGO_PARAMS` 浅拷贝**：`RadarConfig` 用 `.copy()`（L605），内层算法 dict 全局共享 → 运行中改参数污染默认值（bug，须改 `deepcopy`）；
8. **双 DBF 缓存**：paper 路径用 `_dbf_sv_cache/_dbf_angles_cache`，dubhe/optimized 用 `_dbf_sv/_dbf_angles`，`_reset_dubhe_state` 只重置后者 —— 重构时统一；
9. **遗留死代码**：`map_to_2d_grid`（唯一调用被注释）、`_seat_patches`（从未赋值）、`_occ_select_all/clear_all/front`（空实现）、`RadarConfig.playback_file_list`（类级共享可变默认值）；
10. **平台耦合**：TTS 通过 `subprocess` 调 PowerShell `System.Speech`（Windows only），录制语音播报文案硬编码（"开始采样"/"结束采样"）。

---

## 2. 硬编码清单 → 配置映射（第二阶段配置化的依据）

### 2.1 路径类

| # | 位置（行号） | 硬编码 | 配置项（建议） |
|---|---|---|---|
| P1 | 447 / 5495 | `"./model/epoch-25-val-f1-100.0-sp-100.0.tflite"`（默认 OA 模型） | `paths.oa_model`（回退到 algo_params.oa_model_path） |
| P2 | 611 / 618 | `data_save_dir="./data"`、`recent_save_dirs=["./data"]` | `paths.data_dir` |
| P3 | 5475–5482 | 回放导出目录 = `__file__ 目录 / {playback_name}` | `paths.playback_export_dir` |
| P4 | 5190 | 启动时 `os.makedirs(data_save_dir)` | 随 P2 |
| P5 | 54 | `CONFIG_FILE = {__file__目录}/config_save.json` | 保留，但基准改为项目根定位函数 |

### 2.2 协议/通信类

| # | 位置 | 硬编码 | 配置项（建议） |
|---|---|---|---|
| C1 | 535–537 | 帧魔数 `FF 00 FF 00` / `F0 00 F0 00`、`FT_LEN=32`、帧长公式 | `protocol.start_sign/stop_sign/ft_len`（hex 字符串） |
| C2 | 540–543 | `update_protocol` 全局类状态 | 改为实例化 `RadarProtocol(cfg)`，消除全局 |
| C3 | 587–588 | `udp_ip=127.0.0.1`、`udp_port=55555` | 已有配置项（保留默认值即可） |
| C4 | 584–597 | 连接模式/串口/CAN 参数 | 已在 `_SAVE_FIELDS`，无需改 |
| C5 | 4999 | `connection_mode` 下拉可选值 `('UDP','BD_UDP','BD_CAN','SERIAL','CAN','PLAYBACK')` | `comm.modes`（若需增删模式） |

### 2.3 GUI 外观/行为类

| # | 位置 | 硬编码 | 配置项（建议） |
|---|---|---|---|
| G1 | 5188 | 窗口 `geometry("1300x850")` | `gui.window_size="1300x850"` |
| G2 | 4085 | `state_colors = {0:'green',1:'gold',2:'red'}` | `gui.colors.state` |
| G3 | 4303–4342 | 座位椭圆三态颜色/线宽 | `gui.colors.seat` |
| G4 | 4572–4810 大量 | 面板背景 `#f0f0f0`、按钮 `#cfc/#fcc/#ddd` 等 | `gui.colors.panel`（主题 dict，默认值 = 现状） |
| G5 | 4578 | 标题 `"Radar V22 (Fixed)"` | `gui.title` |
| G6 | 631–634 | `imaging_grid`：`linspace(-2,2,37)`/`linspace(-3,-0.1,30)` | `gui.plot.grid`（与 plot extent 联动） |
| G7 | 3649 / 459 | 热力图背景色 `#f2f2f2` | `gui.colors.heatmap_bg`（已在 algo_params，可上移） |
| G8 | 5055–5071 / 5146 | TTS 文案与启用开关 | `gui.speech.enabled/text`（模板） |

### 2.4 录制命名/业务规则类（ControlPanel 内，最典型的硬编码业务逻辑）

| # | 位置 | 硬编码 | 配置项（建议） |
|---|---|---|---|
| R1 | 4902–4912 | `_gen_base_filename` 拼名规则（人员/脚坑/位置/姿势/动作/时间） | `recording.filename_template`（f-string 模板） |
| R2 | 4644 | 单动作时长默认 `"10"` 秒 | `recording.default_action_time` |
| R3 | 4845–4865 | 动作解析、总时长估算公式 | 随 R1 模板化 |
| R4 | 4922–4941 | 重名自动加后缀规则 `_1`/`_2` | 保留默认，可配置分隔符 |

### 2.5 算法参数类（已部分配置化，需完善）

| # | 位置 | 现状 | 改进 |
|---|---|---|---|
| A1 | 154–529 | `DEFAULT_ALGO_PARAMS` 9 套算法魔数 | 已可被 `config_save.json.algo_params` 覆盖；**补 JSON Schema 校验**（类型/范围），防手改配置出错 |
| A2 | 605 | `algo_params = DEFAULT_ALGO_PARAMS.copy()` 浅拷贝 | 改 `deepcopy`（修复跨实例污染） |
| A3 | 5200–5203 | OA 缓冲 `deque(maxlen=8)`、BD 缓冲 `maxlen=3000` | 收进 `algo_params['RA-OCCUPANCY'].oa_buffer_len` 等 |
| A4 | 447–508 | `oa_mean_threshold`、`heatmap_update_stride_combined` 等 | 已配置化 ✓（保持） |

---

## 3. 目标架构（最终形态）

```
app/
├── __init__.py
├── config.py            # NpEncoder + 项目根定位 + CONFIG_FILE + save/load_config
│                        #   + RadarConfig + ANTENNA_LAYOUTS + DEFAULT_ALGO_PARAMS
├── protocol.py          # RadarProtocol(实例化) + ConfigAdapter + FrameParser 注册表(V1/V2 插槽)
├── data_processing.py   # FixedBuffer + BackgroundRemoval + RadarDataManager
├── algorithms.py        # AlgorithmProcessor(整类) + calculate_mdl_asc_local + apply_range_bin_selection
├── detectors.py         # SeatOccupancyDetector + RAOccupancyDetector + seat_config() 共享 helper
├── sources.py           # BaseSource(抽象) + LiveRadarSource + FilePlaybackSource
├── ra_occ_model.py      # TF Lite 懒加载/推理/量化适配 + BD 采样导出（从 App 抽出）
├── gui/
│   ├── __init__.py
│   ├── theme.py         # 颜色/尺寸/文案 主题（读 gui.* 配置）
│   ├── plot_panel.py    # PlotPanel
│   ├── control_panel.py # ControlPanel（命名规则改读模板）
│   └── dialogs.py       # AlgoSettingsDialog + SeatConfigDialog
├── gui_app.py           # App（编排层，拆后约 200 行）
└── main.py              # 薄入口（原 __main__ 块）

gui_main.py              # 拆完后删除，或保留为 `from app.main import main; main()` 兼容 shim
config_save.json         # 升级为 v2 分区：{version, app, gui, comm, protocol, paths, recording, radar_config, algo_params}
```

依赖方向（无环）：

```
main → gui_app → {gui/*, sources, algorithms, detectors, ra_occ_model, config, data_processing}
algorithms → {config, data_processing, util, breathe}
sources → {protocol, config, CAN_data_listen, UDP_data_listen, device_management}
protocol → (仅 numpy/struct)
config → (无内部依赖)
```

---

## 4. 分阶段执行计划

> 每个阶段末尾：`git commit`。阶段 1 是纯搬迁；阶段 2 是配置化；阶段 3 是解耦/优化；阶段 4 是回归。

### 阶段 0：基线准备（0.5 天）

| 步骤 | 动作 | 验证 |
|---|---|---|
| 0.1 | 新建分支 `refactor/gui-decouple`；确认 `git status` 干净 | `git status` |
| 0.2 | 用当前版本 GUI 录制 1 个含空座/成人/儿童场景的 `.bin` 基准样本（放 `data/baseline/`），并导出对应点云 JSON | 文件存在且非空 |
| 0.3 | 写验证脚本 `tools/verify_imports.py`：遍历 `app/**/*.py` 做 `py_compile` + import（用 py310） | 全绿 |
| 0.4 | 写回放对比脚本：同一 `.bin` 分别过"重构前/后"代码，逐帧对比 `(tx,rx,cir)` 与 RA 热力图输出（允许 float 容差 1e-6） | 输出 diff = 0 |

### 阶段 1：纯搬迁拆包（行为不变）（2–3 天）

> 顺序：叶子 → 根。每步"原样搬迁 + 修正 import + 模块级验证"，**不改任何逻辑**。
> 通用验证：`py310 -c "import app.<模块>"`（在项目根目录执行）。

| 步骤 | 动作 | 依赖前提 | 专项验证 |
|---|---|---|---|
| 1.1 | 建 `app/config.py`：NpEncoder、`PROJECT_ROOT` 定位函数、CONFIG_FILE、save/load_config、RadarConfig（**含 deepcopy 修复**，见 3.2）、ANTENNA_LAYOUTS、DEFAULT_ALGO_PARAMS | 无 | `save/load_config` 往返一致；两个布局可 load |
| 1.2 | 建 `app/protocol.py`：RadarProtocol（暂保持类方法/全局态，第二阶段改实例）、ConfigAdapter | config | 用基准 .bin 回放验证 `parse_frame` 输出一致 |
| 1.3 | 建 `app/data_processing.py`：FixedBuffer、BackgroundRemoval、RadarDataManager | config, protocol | 构造 + `process_frame` 冒烟 |
| 1.4 | 建 `app/algorithms.py`：AlgorithmProcessor 整体 + calculate_mdl_asc_local + apply_range_bin_selection | config, data_processing, util, breathe | 离线构造 + 9 条流水线各跑 1 帧（无 GUI） |
| 1.5 | 建 `app/detectors.py`：两个检测器 + seat_config() 去重 | 无 | 构造 + 假数据 process |
| 1.6 | 建 `app/sources.py`：LiveRadarSource、FilePlaybackSource（依赖契约不变） | protocol, config | PLAYBACK 模式跑基准 .bin |
| 1.7 | 建 `app/ra_occ_model.py`：从 App 抽 `_ra_occ_*`（加载/推理/归一化/BD 采样导出） | config | 无模型时优雅降级；有模型时输出一致 |
| 1.8 | 建 `app/gui/theme.py`：把 §2.3 的颜色/尺寸/文案**抽成默认 dict（值=现状）**，暂不读配置 | 无 | 界面截图对比无视觉差异 |
| 1.9 | 建 `app/gui/plot_panel.py`、`app/gui/control_panel.py`、`app/gui/dialogs.py` | config, detectors, theme | 窗口能弹出；控件行为一致 |
| 1.10 | 建 `app/gui_app.py`：App 整体搬迁；建 `app/main.py` 薄入口；`gui_main.py` 改为 shim（`from app.main import main; main()`） | 全部 | `python gui_main.py` 启动正常 |
| 1.11 | 全量回归：回放对比脚本输出 diff=0；GUI 各 mode 手动点检 | — | 见阶段 0.4 |

**阶段 1 验收**：`gui_main.py` 变成 3 行 shim；9 个新模块均可独立 import；行为与重构前完全一致。

### 阶段 2：硬编码 → 配置化（2–3 天）

> 原则：**配置项默认值 = 现状**，未配置时行为不变；新增配置项必须先写默认值。

| 步骤 | 动作 | 涉及硬编码 |
|---|---|---|
| 2.1 | `config_save.json` 升级 v2 分区（`app/gui/comm/protocol/paths/recording/radar_config/algo_params`）；`save_config/load_config` 支持分区读写，旧文件自动迁移 | P1–P5 |
| 2.2 | 路径类接入：OA 模型路径、数据目录、回放导出目录全部走 `paths.*`，解析统一基于 `PROJECT_ROOT` | P1–P5 |
| 2.3 | 协议类接入：`RadarProtocol` 改为实例化对象，帧魔数/FT_LEN 从 `protocol.*` 读取；删掉 `update_protocol` 全局类状态（改由 source 构造时传入实例）；`RadarProtocol.update_protocol(ft_len)` 的所有调用点同步改 | C1–C2 |
| 2.4 | GUI 外观接入：`theme.py` 从 `gui.*` 配置加载，PlotPanel/ControlPanel 全部颜色、窗口尺寸、标题、TTS 文案改读 theme | G1–G8 |
| 2.5 | 录制命名规则接入：`_gen_base_filename` 改为读 `recording.filename_template`（示例模板 `"{person}_{area}_{position}_{pose}_{action}_{time}.bin"`），解析函数从模板生成；旧硬编码逻辑作为默认模板 | R1–R4 |
| 2.6 | 算法参数补 Schema：`config.py` 提供 `validate_algo_params()`，启动时校验类型/数值范围（如 `heatmap_update_stride_combined>=1`、`azimuth_num>0`），非法值打警告并回落默认 | A1–A4 |
| 2.7 | 为 `algo_params` 增加运行时只读视图（`get_algo(name)` 返回 deepcopy），杜绝外部直接改默认值 | A2 |

**阶段 2 验收**：改 `config_save.json` 的窗口尺寸/颜色/帧魔数/命名模板后**重启即生效**，无需改代码；默认配置下行为与阶段 1 完全一致。

### 阶段 3：解耦与架构优化（3–5 天）

| 步骤 | 动作 | 说明/风险 |
|---|---|---|
| 3.1 | **FrameParser 注册表**：`protocol.py` 定义 `ParsedFrame` dataclass + `FrameParser` 抽象基类 + `UwbV1Parser`（现格式）+ `register()/parse_frame()` 入口；sources 改为按 `parser.frame_len` 切帧 | 为新下位机帧格式留插槽；**行为不变**（V1 逻辑原样迁入） |
| 3.2 | **输出契约 TypedDict**：为 9 条流水线的 `step_*` 返回值定义 `TypedDict`（如 `RAOccupancyResult`），PlotPanel 按契约读 key，杜绝 dict key 静默改名 | 回归风险高，需回放对比 |
| 3.3 | **数据源抽象**：`sources.py` 定义 `BaseSource`（running/start/stop/get_batch_frames/get_rec_status/get_progress），两个 source 实现之；App 只依赖抽象 | 契约固化 |
| 3.4 | **App 瘦身**：BD 采样导出（`_bd_*`）、OA 后处理（`_ra_occ_*`）全部收进 `ra_occ_model.py` 或新 `app/services/`；App 只留编排 | 大段移动，逐方法验证 |
| 3.5 | **死代码清理**：删 `map_to_2d_grid`、`_seat_patches`、`_occ_select_all/clear_all/front`、`RadarConfig.playback_file_list` 类级默认（改为 `field(default_factory=list)`） | 先 grep 确认无引用 |
| 3.6 | **重复预处理消重**：`step_point_cloud_optimized` / `step_angle_spectrum_view` / `step_angle_spectrum_raw` 三段约 100 行相同预处理抽公共函数 | **回归风险最高**，必须回放对比 |
| 3.7 | **统一双 DBF 缓存**：`_dbf_sv_cache` 与 `_dbf_sv` 合并，`_reset_dubhe_state` 统一重置 | 行为核对 |
| 3.8 | **性能优化**：steering vector / CFAR 窗口等只算一次的缓存复用；`get_all_snapshot_as_array` 避免每帧全量重建；热力图更新节流逻辑复核 | 对比重构前后 FPS |
| 3.9 | **类型注解补全**：公共接口（sources/protocol/detectors/config）加类型注解与 docstring；算法内部方法可后置 | 用 `mypy` 抽查 |
| 3.10 | **TTS 解耦**：`gui/speech.py` 封装 `SpeechAnnouncer`（默认 PowerShell 实现，可换 pyttsx3），文案走配置模板 | 平台耦合收口 |

**阶段 3 验收**：`AlgorithmProcessor` 不再被 GUI import；App < 300 行；死代码清零；回放对比 diff=0；FPS 不下降。

### 阶段 4：回归与收尾（1 天）

| 步骤 | 动作 | 验证 |
|---|---|---|
| 4.1 | 全量 `py_compile` + import 测试（py310） | 全绿 |
| 4.2 | PLAYBACK 回放基准 .bin：逐帧 `(tx,rx,cir)` + RA 热力图对比重构前 | diff=0 |
| 4.3 | GUI 手动点检：6 种连接模式入口、11 种绘图 mode、录制/回放/导出、OA 模型开关、座位配置对话框 | 全部可用 |
| 4.4 | 真机冒烟（如有硬件）：CAN 模式接收 + 录制 + 座位显示 | 正常 |
| 4.5 | 更新 README（新结构/配置说明）；删除 `gui_main.py` shim 或保留 | 文档一致 |
| 4.6 | 提交并打 tag（如 `refactor-v2`） | — |

---

## 5. 风险与回退

| 风险 | 缓解 |
|---|---|
| 拆包后 import 断 | 每步立即 import 验证；小步 commit |
| 行为漂移（尤其 3.6 三段共享预处理） | 阶段 0 先录基准 .bin，重构后回放对比 |
| 路径漂移（`__file__`/CWD） | 阶段 1 只搬不改；`PROJECT_ROOT` 统一基准在阶段 2 一次性完成 |
| 循环导入 | 严格按叶子→根顺序；`TYPE_CHECKING` 注解 |
| 配置化引入新 bug | 默认值=现状；未配置时走默认分支；2.6 的 Schema 校验兜底 |
| 进度失控 | 每阶段一个 commit；任何时刻 `git checkout refactor/gui-decouple~N` 回退 |

---

## 6. 工作量预估

| 阶段 | 内容 | 预估 |
|---|---|---|
| 0 | 基线 | 0.5 天 |
| 1 | 纯搬迁拆包 | 2–3 天 |
| 2 | 配置化 | 2–3 天 |
| 3 | 解耦/优化 | 3–5 天 |
| 4 | 回归 | 1 天 |
| **合计** | | **8.5–12.5 天** |

> 若时间紧，可裁剪：阶段 1 + 阶段 2（配置化）即可解决"冗杂 + 硬编码"两大痛点，阶段 3 的 3.6/3.7/3.8 可延后。
