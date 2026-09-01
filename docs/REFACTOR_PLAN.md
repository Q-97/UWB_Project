# gui_main.py 重构规划（5769 行 → 模块化）

> 目标：拆分 5769 行的 `gui_main.py` 上帝文件 + 为"新下位机解帧方式"预留协议扩展点。
> 分支：`refactor/gui-modularization`（基线 = 学长 `e2503bb`，随时可 `git checkout 样本采集` 回退）。
> 本文档基于 7 段代码结构分析（2025 年 7 段并行审计）编写。

---

## 1. 现状全景

### 1.1 文件结构地图（gui_main.py）

| 行号 | 内容 | 拆往 | 备注 |
|---|---|---|---|
| 1–36 | import + matplotlib 全局副作用 | 分散 | `matplotlib.use('TkAgg')`、SimHei rcParams 是导入即生效的全局副作用 |
| 41–134 | NpEncoder / save_config / load_config / apply_range_bin_selection | `utils.py` / `config.py` | save/load 与 RadarConfig 鸭子类型耦合 |
| 139–529 | ANTENNA_LAYOUTS / DEFAULT_ALGO_PARAMS | `config.py` | 9 套算法默认参数 + 2 套天线布局 |
| 534–554 | **RadarProtocol（解帧入口）** | **`protocol.py`** | 无版本字段、无 CRC，新解帧方式的插入点 |
| 555–572 | ConfigAdapter | `protocol.py` | 桥接 RadarConfig → 帧服务器字段名 |
| 573–643 | RadarConfig | `config.py` | dataclass，25+ 字段 |
| 644–759 | FixedBuffer / BackgroundRemoval / RadarDataManager | `data_processing.py` | 数据层 |
| 761 | calculate_mdl_asc_local | `algorithms.py` | 算法层内部函数 |
| 774–3092 | **AlgorithmProcessor（2320 行）** | `algorithms.py` | 整个类整体搬迁，不可按方法切 |
| 3093–3331 | SeatOccupancyDetector / RAOccupancyDetector | `detectors.py` | 零 GUI 依赖，拆包零风险 |
| 3332–3482 | LiveRadarSource / FilePlaybackSource | `sources.py` | 依赖 RadarProtocol（须一并迁出） |
| 3483–4346 | PlotPanel | `gui/plot_panel.py` | 11 种 mode |
| 4347–4569 | SeatConfigDialog | `gui/dialogs.py` | |
| 4570–5185 | ControlPanel | `gui/control_panel.py` | ~45 个方法 |
| 5186–5767 | **App（上帝类）** | `app.py` | 编排 12+ 子系统 |
| 5768–5769 | `__main__` 入口 | `main.py` | 薄入口 |

### 1.2 关键耦合事实（拆包前必须知道）

1. **依赖方向整体无环**：`ConfigAdapter→RadarProtocol`、`RadarDataManager→{FixedBuffer, BackgroundRemoval, RadarConfig}`、`App→{全部}`，子对象之间**从不直接互引**，全部经 App 中转。这是最容易拆的形态。
2. **循环导入的唯一雷区**：`RadarConfig`、`RadarDataManager`、`calculate_mdl_asc_local` 与 AlgorithmProcessor 同文件。若 `algorithms.py` 反向 `import gui_main` 即死循环 → **必须先把公共符号下沉到独立模块**，或对类型注解用 `TYPE_CHECKING`。
3. **AlgorithmProcessor 必须整体搬迁**：中段调用 8 个段外兄弟方法，且 `_angle_cfar_detect`/`_find_angle_peaks` 被后段 `_dbf_estimate` 反向调用，任何按方法切分都会断。
4. **App 的鸭子类型契约（属性名不能改）**：`source.running`、`source.data_queue.empty()`、`source.measured_fps/playback_fps`、`get_rec_status()`、`get_progress()`、`get_batch_frames()`；`cbs` 回调 key 全集 = `{start, stop, rec_start, rec_stop, update_layout}`；`p['occupancy_config']` 隐式键。
5. **GUI 与算法只在契约层耦合**：`PlotPanel.update_data(mode, d)` 按 mode 硬编码读 dict key；`App.loop` 按 `config.current_algo` 字符串分发。算法模块**没有任何 tkinter/matplotlib 调用**。
6. **3 处隐藏路径**（拆到子目录会漂移）：
   - `CONFIG_FILE` 基于 `__file__` 绝对路径（拆包后基准变）
   - `oa_model_path = "./model/…tflite"` 相对 CWD
   - `data_save_dir = "./data"` 相对 CWD
7. **`DEFAULT_ALGO_PARAMS` 是浅拷贝**：`RadarConfig` 用 `.copy()`，内层算法 dict 全局共享 → 运行中改参数会污染默认值。拆包时改 `deepcopy`。
8. **双 DBF 缓存**：paper 路径用 `_dbf_sv_cache/_dbf_angles_cache`，dubhe/optimized 用 `_dbf_sv/_dbf_angles`，`_reset_dubhe_state` 只重置后者——重构时统一。
9. **遗留死代码**（可清理）：`map_to_2d_grid`（唯一调用被注释）、`_seat_patches`（从未赋值）、`_occ_select_all/clear_all/front`（空实现）、`RadarConfig.playback_file_list`（类级共享可变默认值）。

---

## 2. 目标模块结构

```
uwb-occupancy-detection/
├── main.py                 # 入口（原 __main__ 块：建 Tk → App → WM_DELETE_WINDOW 回调）
├── app.py                  # App 编排层（原 5186–5767，最后拆）
├── config.py               # RadarConfig + ANTENNA_LAYOUTS + DEFAULT_ALGO_PARAMS
│                           #   + CONFIG_FILE/_SAVE_FIELDS/save_config/load_config
├── utils.py                # NpEncoder + apply_range_bin_selection + 通用小工具
├── protocol.py             # RadarProtocol + ConfigAdapter + FrameParser 注册表（新解帧插槽）
├── data_processing.py      # FixedBuffer + BackgroundRemoval + RadarDataManager
├── algorithms.py           # AlgorithmProcessor（整个类）+ calculate_mdl_asc_local
├── detectors.py            # SeatOccupancyDetector + RAOccupancyDetector + seat_config() 共享helper
├── sources.py              # LiveRadarSource + FilePlaybackSource
├── ra_occ_model.py         # TF Lite 懒加载/推理/量化适配（从 App 抽出）
├── gui/
│   ├── __init__.py
│   ├── plot_panel.py       # PlotPanel
│   ├── control_panel.py    # ControlPanel
│   └── dialogs.py          # AlgoSettingsDialog + SeatConfigDialog
│
├── 现有独立模块（不动）: CAN_data_listen.py / UDP_data_listen.py / device_management.py
│                        / radar_config.py / util.py / breathe.py / canfd.py / zlgcan.py
└── gui_main.py             # 全部拆完后删除（或保留为兼容 shim）
```

依赖方向（无环）：
```
main → app → {gui/*, sources, algorithms, detectors, ra_occ_model, config, data_processing}
algorithms → {config, data_processing, utils, util, breathe}
sources → {protocol, config, CAN_data_listen, UDP_data_listen, device_management}
protocol → (无内部依赖，仅 numpy/struct)
config → utils
```

---

## 3. 拆包顺序（每步可验证、可回退）

> 原则：**先搬后改**——每步只做"原样搬迁 + 修正 import"，行为不变；逻辑重构放到第 2 阶段。
> 每步验证：`python -c "import <模块>"` + 无硬件冒烟（PLAYBACK 模式跑一个已录制的 .bin）。

| 步骤 | 动作 | 依赖前提 | 验证 |
|---|---|---|---|
| 0 | 提交当前基线（已就绪：分支 `refactor/gui-modularization`，HEAD=17b09d9） | — | `git status` 干净 |
| 1 | 建 `utils.py`：NpEncoder、apply_range_bin_selection | 无 | import + 单测参数 |
| 2 | 建 `protocol.py`：RadarProtocol + ConfigAdapter（含新 FrameParser 抽象基类 + V1 实现，见 §5） | 无 | 用现成 .bin 回放验证 parse |
| 3 | 建 `config.py`：RadarConfig + ANTENNA_LAYOUTS + DEFAULT_ALGO_PARAMS + save/load_config + CONFIG_FILE + _SAVE_FIELDS | utils | load/save 往返一致 |
| 4 | 建 `data_processing.py`：FixedBuffer + BackgroundRemoval + RadarDataManager | config, protocol | 构造 + process_frame 冒烟 |
| 5 | 建 `algorithms.py`：AlgorithmProcessor 整体 + calculate_mdl_asc_local | config, data_processing, utils, util, breathe | 离线构造 + step_waveform |
| 6 | 建 `detectors.py`：两个检测器 + seat_config() helper（消重） | 无（仅 numpy/time/deque） | 构造 + process 假数据 |
| 7 | 建 `sources.py`：两个数据源 | protocol, config | PLAYBACK 模式跑通 |
| 8 | 建 `ra_occ_model.py`：从 App 抽出 `_ra_occ_*` 模型方法 | config | 无模型时优雅降级 |
| 9 | 建 `gui/` 包：plot_panel.py / control_panel.py / dialogs.py | config, detectors | 窗口能弹出 |
| 10 | 建 `app.py`：App 整体搬迁 | 全部 | PLAYBACK 全流程 |
| 11 | 建 `main.py`：入口；`gui_main.py` 删空或保留 shim | app | `python main.py` |
| 12 | 提交 + 全量回归（PLAYBACK 对比重构前后输出） | — | 行为 diff |

---

## 4. 第二阶段：行为不变的加固（拆完后做）

1. `DEFAULT_ALGO_PARAMS` 浅拷贝 → `deepcopy`（修复跨实例参数污染）。
2. **契约固化**：为各 `step_*` 输出定义 `TypedDict`（`PointCloudResult`、`RAOccupancyResult`…），杜绝 GUI 侧 dict key 静默改名。
3. **消重**：`step_point_cloud_optimized` / `step_angle_spectrum_view` / `step_angle_spectrum_raw` 三段 ~100 行相同预处理抽公共函数（**回归风险最高，需对比输出**）。
4. **统一双 DBF 缓存**：`_dbf_sv_cache` 与 `_dbf_sv` 合并。
5. `dm.snapshots_data` 内部 dict 直连 → 提供查询接口。
6. 算法参数字符串 key 读取 → 定义参数 dataclass（可选，工作量较大）。
7. `PlotPanel` 暴露 `reset_history()`，替代 App 直摸私有属性 `pc_history` / `ra_occ_oa_filter_history`。
8. 清理死代码（§1.2 第 9 条）。
9. `CONFIG_FILE` / `oa_model_path` / `data_save_dir` 路径基准归一（基于模块 `__file__` 或显式配置）。

---

## 5. 新解帧方式设计（第二个目标）

### 5.1 现状

```python
# 当前协议（RadarProtocol V1）：138 字节固定帧
# [4B 魔数 ff 00 ff 00][2B tx,rx][128B I/Q 交织 int16][4B 尾魔数]
# 校验：仅"长度精确匹配 + 帧头魔数"，无 CRC、无版本字段
```

关键障碍：**帧内没有协议版本号**，新旧帧混流时无法自动区分。

### 5.2 方案：FrameParser 注册表（协议插件化）

```python
# protocol.py —— 本次重构即建好插槽
@dataclass
class ParsedFrame:
    tx: int
    rx: int
    cir: np.ndarray          # complex64, len FT_LEN
    raw: bytes

class FrameParser(ABC):
    name: str
    frame_len: int           # 该协议的单帧字节长（供 source 定长切帧）
    @abstractmethod
    def can_parse(self, frame: bytes) -> bool: ...
    @abstractmethod
    def parse(self, frame: bytes) -> ParsedFrame: ...

class UwbV1Parser(FrameParser): ...   # 现有 138B 格式，原样迁入
# class UwbV2Parser(FrameParser): ... # 新下位机：待帧格式定稿后实现

PARSERS: list[FrameParser] = [UwbV1Parser()]
def register(parser): PARSERS.append(parser)
def parse_frame(frame: bytes) -> ParsedFrame:
    for p in PARSERS:
        if p.can_parse(frame):
            return p.parse(frame)
    raise ProtocolError(...)
```

### 5.3 给新下位机的帧格式建议（下位机代码未写，正好现在定）

- **必须加 1 字节协议版本字段**（V1 魔数头后可扩展），这样 `can_parse` 只需查版本号，source 也能按 `frame_len` 切帧，新旧帧**可混流自动识别**；
- **必须加 CRC/校验和**（当前 V1 无校验，错帧静默）；
- 建议在帧头加**显式长度字段**（`frame_len`），source 不再依赖全局 `RadarProtocol.FRAME_LEN`（当前 `update_protocol(ft_len)` 是全局可变状态，拆包后要消除）；
- 若新下位机**无法**加版本字段 → 退化为方案 B：`config.connection_mode` 或新增 `config.protocol` 字段手动选协议（不改 source 逻辑，只在启动时选 PARSERS）。

### 5.4 改动面

| 位置 | 改动 |
|---|---|
| `protocol.py` | 注册表 + V1 实现（原样迁入） |
| `sources.py` | `_io_loop`/`_play_loop` 从 `RadarProtocol.parse_frame` 改为 `parse_frame()` 注册表入口；切帧长度改为按当前协议的 `frame_len`（替代全局 `FRAME_LEN`） |
| `config.py` | （方案 B 时）新增 `protocol` 字段 |
| `app.py` | `start()` 里按需选择协议，不再调 `RadarProtocol.update_protocol` 改全局 |

---

## 6. 风险与回退

| 风险 | 缓解 |
|---|---|
| 拆包后 import 断 | 每步后立即 import 验证；小步提交（`git commit` 每步一个） |
| 行为漂移（尤其 3 段共享预处理） | 第 2 阶段前先录 1 个 .bin 基准样本，重构后 PLAYBACK 对比输出 |
| 路径漂移（`__file__`/CWD） | 第 1 阶段只搬不改；路径归一放第 2 阶段并单独验证 |
| 循环导入 | 严格按 §3 顺序拆（叶子→根）；`TYPE_CHECKING` 注解 |
| 进度失控 | 每阶段结束一个 commit；任何时刻 `git checkout 样本采集` 回到学长原版 |

## 7. 建议的执行节奏

1. **第 1 阶段（纯搬迁，行为不变）**：建议一次做完 1–7 步（utils→protocol→config→data_processing→algorithms→detectors→sources），这是纯体力活 + import 修正，每步 commit；
2. **第 1.5 阶段**：8–11 步（ra_occ_model→gui→app→main），完成后 `gui_main.py` 退役；
3. **第 2 阶段（加固）**：按优先级做 §4 各项；
4. **第 3 阶段（新解帧）**：下位机帧格式定稿后，实现 `UwbV2Parser` 一行注册即可，其余代码零改动。
