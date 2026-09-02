# 上位机减负清单（仅保留 RA-OCCUPANCY）

> 依据：用户确认生产只保留 **RA-OCCUPANCY**（占用检测）。本文档用 AST 可达性分析
> （`tools/analyze_slim.py`）从 `AlgorithmProcessor.step_ra_occupancy` 出发求依赖闭包，
> 列出精确删除候选（文件 + 行号 + 预计行数）与配套改造项。**未确认前不执行。**

## 0. 现状体量（app/ 共 ~8000 行，AST 行号为准确值）

| 文件 | 当前行数(AST) | 瘦身后估计 | 预计节省 |
|---|---|---|---|
| algorithms.py | ~2400 | ~450 | **~1950** |
| util.py | 845 | 0（整体删除） | ~845 |
| breathe.py | 91 | 0（整体删除） | ~91 |
| plot_panel.py | ~730 | ~330 | ~400 |
| config.py | ~660 | ~430 | ~230 |
| detectors.py | ~240 | ~110 | ~130 |
| control_panel.py | ~530 | ~470 | ~60 |
| gui_app.py | ~250 | ~200 | ~50 |
| **合计** | **~8000** | **~4300** | **~3700（≈45%）** |

## 1. algorithms.py — 删除 34 个不可达方法（约 1950 行）

### 保留（RA-OCCUPANCY 闭包，共 6 方法 + 基建）

| 方法 | 行号 |
|---|---|
| `__init__` / `_reset_algo_state` / `_reset_dubhe_state` | L131-152 |
| `_get_capon_antenna_x` | L643-659 |
| `_seat_ra_energy` | L1203-1219 |
| `_ra_to_cartesian` | L1221-1243 |
| `step_ra_occupancy` | L1245-1315 |
| `_init_capon_steering` | L1910-1929 |
| `_compute_ra_heatmap` | L1931-2001 |
| 模块级 `apply_range_bin_selection` | 保留 |

### 删除候选（不可达，按块给出行号）

| 块 | 行号 | 行数 |
|---|---|---|
| `_init_music_if_needed` + `_run_2d_music_mdl_internal` | L154-306 | ~153 |
| `step_2d_music` | L308-403 | ~96 |
| `step_waveform` | L405-420 | ~16 |
| CFAR/AoA 族：`perform_ca_cfar_2d` `estimate_aoa` `calculate_2d_fft_aoa` `calculate_music_2d_aoa` `calculate_fft_aoa` `calculate_music_aoa` `calculate_capon_aoa` | L423-709 | ~287 |
| `step_point_cloud` + `_prep_doppler_cube` + `_power_map_selected` + `step_point_cloud_optimized` | L711-940 | ~230 |
| `step_angle_spectrum_view` / `step_angle_spectrum_raw` | L942-1076 | ~134 |
| `step_point_cloud_ra_cfar` | L1078-1181 | ~104 |
| `step_ra_heatmap_view` | L1183-1201 | ~19 |
| 1D/角度谱族：`perform_1d_cfar_min` `_bartlett_spectrum` `_compute_angle_response` `_angle_cfar_detect` `_find_angle_peaks` `_fft_angle_response` `_dbf_angle_response` `_music_angle_response` `_capon_angle_response` | L1319-1598 | ~280 |
| `step_point_cloud_paper` | L1600-1852 | ~253 |
| `_subbin_refine` + `_init_dbf_steering` + `_build_dbf_steering` | L1856-1908 | ~53 |
| `_two_pass_cfar` + `_dbf_estimate` | L2003-2230 | ~227 |
| `step_point_cloud_dubhe` | L2232-2380 | ~149 |
| 模块级 `calculate_mdl_asc_local` | 删除 | ~12 |

> ⚠️ 其中 `_prep_doppler_cube`/`_power_map_selected`/`_build_dbf_steering` 是上一轮刚抽的
> 公共方法，此处删除是因为其唯一调用方（点云/角度谱流水线）全部移除——**不是反悔重构，
> 是功能裁剪的必然结果**。

### import 修剪
- 删除 `from app import util`、`from app.breathe import find_breathing_feature`（闭包无引用）
- 视保留代码实际使用裁剪 `scipy.signal`/`ndimage`/`chebwin` 等（execution 时逐个核对）

## 2. util.py（845 行）— 整体删除

AST + grep 确认：`util.*` 仅被 algorithms.py 不可达方法使用
（`create_virtual_array_mapping`←`_init_music_if_needed`、`find_xy_peaks_from_image`/
`find_peak_indices_np`/`threshold_func`←2D-MUSIC 族），其余 util 函数本就无引用。
瘦身后 app/ 内无任何 `util` 引用 → 删除 `app/util.py`。

## 3. breathe.py（91 行）— 整体删除

`find_breathing_feature` 的 6 处调用（L387/784/927/1168/1838/2367）全部位于待删方法内，
RA-OCCUPANCY 闭包不使用呼吸特征 → 删除 `app/breathe.py`。

## 4. config.py — DEFAULT_ALGO_PARAMS 收窄（约省 230 行）

- 算法默认参数仅保留 `"RA-OCCUPANCY"` 与 `"SEAT-OCCUPANCY"`（RA 座位区/阈值依赖），
  删除 PLOT / 2D-MUSIC / POINT-CLOUD / POINT-CLOUD-OPTIMIZED / POINT-CLOUD-DUBHE /
  POINT-CLOUD-PAPER / RA-CFAR 七个 dict
- `validate_algo_params()` 校验规则同步收窄到 RA-OCCUPANCY/SEAT-OCCUPANCY
- `load_config()` 对已保存 v2 配置中多余 algo key 自动忽略（现有逻辑已天然兼容）

## 5. detectors.py — 删除 SeatOccupancyDetector（约省 130 行）

- 仅 RAOccupancyDetector 供 RA-OCCUPANCY 使用；SeatOccupancyDetector 只服务于被删的点云模式
  （gui_app 中 3 处实例化 + App.loop 点云分支）→ 删除类并同步 gui_app 引用

## 6. GUI 收窄

| 文件 | 动作 |
|---|---|
| plot_panel.py | `init_layout`/`update_data` 仅保留 `RA-OCCUPANCY` 分支（含座位椭圆/田字格状态面板/热力图），删除 PLOT/2D-MUSIC/POINT-CLOUD*/RA-CFAR/RA-HEATMAP/ANGLE-SPECTRUM/AS-RAW 分支（约省 400 行） |
| gui_app.py | `App.__init__` 初始布局 `"PLOT"` → `"RA-OCCUPANCY"`；`loop()` 删除 step 分发，仅保留 RA-OCCUPANCY + 录制/回放/导出骨架；删除 seat_detector |
| control_panel.py | 算法下拉框值收敛为 `('RA-OCCUPANCY',)` |

## 7. 测试工具配套（否则回归脚本会挂）

| 文件 | 动作 |
|---|---|
| tools/regression_smoke.py | 移除 `step_point_cloud_optimized`/`step_ra_heatmap_view` 调用，改为 RA-OCCUPANCY 全链路（解析→背景去除→step_ra_occupancy→RAOccupancyDetector→OccupancyModelService OA 路径） |
| tools/gui_smoke.py | `_MODES` 收敛为 `['RA-OCCUPANCY']` |
| tools/analyze_slim.py | 保留（减负审计工具） |

## 8. 明确保留（不动的部分）

`config.py`（基础设施）、`protocol.py`、`data_processing.py`、`sources.py`、
`CAN_data_listen.py`、`UDP_data_listen.py`、`device_management.py`、`comm/*`（驱动）、
`radar_config.py`（并入 protocol 属可选）、`ra_occ_model.py`、`gui/` 其余文件、
`gui_main.py` shim、`config_save.json` v2 格式。

## 9. 执行与验证方案（确认后按此进行）

1. 每删一块立即 `python -m py_compile` + 全链 `import gui_main`
2. 主验证：`tools/regression_smoke.py`（改造后 = RA-OCCUPANCY 全链路）+ `tools/gui_smoke.py`
3. 中间产物对比：`RA-OCCUPANCY` 的 `(heatmap, energy, state, oa_status)` 输出在删除前后逐值一致
   （删除只影响未走到的代码，RA 闭包代码逐字保留，理论上零漂移）
4. 每批一个 commit，可随时 `git checkout` 回退
5. 用户真机跑 GUI 目检
