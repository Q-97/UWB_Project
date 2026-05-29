# Dubhe CPD Algorithm User Guide (Dubhe CPD 算法用户指南)

**by Peng Hui** *hui.peng@calterah.com* *SEC System @ Calterah* **User Guide presented to Internal Department** **COMPANY CONFIDENTIAL** *Shanghai, China @2026-03-10* *Classified - No Distribution without Approval @2026, Calterah Semiconductor*

## Version Notes (版本记录)

| Date       | Version | Description                                                 | Author   |
| ---------- | ------- | ----------------------------------------------------------- | -------- |
| 2026-02-09 | V1.0    | Initial Version - CPD Algorithm User Guide                  | Peng Hui |
| 2026-03-09 | V1.1    | Remove redundant sections; update the interface description | Peng Hui |

## Table of Contents (目录)

- [List of Figures](#list-of-figures)
- [List of Tables](#list-of-tables)
- [List of Abbreviations](#list-of-abbreviations)
- [List of Symbols](#list-of-symbols)
- [Chapter 1: Introduction](#chapter-1-introduction)
  - [1.1 Overview](#11-overview)
  - [1.2 System Architecture](#12-system-architecture)
  - [1.3 Key Features](#13-key-features)
  - [1.4 Algorithm Pipeline](#14-algorithm-pipeline)
  - [1.5 File Structure](#15-file-structure)
  - [1.6 Hardware Requirements](#16-hardware-requirements)
  - [1.7 Software Dependencies](#17-software-dependencies)
- [Chapter 2: Configuration File Guide](#chapter-2-configuration-file-guide)
  - [2.1 Overview](#21-overview)
  - [2.2 General Control Parameters](#22-general-control-parameters)
    - [2.2.1 Parameter Details](#221-parameter-details)
  - [2.3 Waveform Parameters](#23-waveform-parameters)
    - [2.3.1 Parameter Details](#231-parameter-details)
  - [2.4 Raw CIR Processing Parameters](#24-raw-cir-processing-parameters)
    - [2.4.1 CIR Combination and Sliding Window](#241-cir-combination-and-sliding-window)
    - [2.4.2 SNR Boost](#242-snr-boost)
  - [2.5 Doppler Processing Parameters](#25-doppler-processing-parameters)
    - [2.5.1 Leakage Offset](#251-leakage-offset)
  - [2.6 Channel Combination Parameters](#26-channel-combination-parameters)
  - [2.7 CFAR Detection Parameters](#27-cfar-detection-parameters)
    - [2.7.1 Doppler Velocity Constraints](#271-doppler-velocity-constraints)
    - [2.7.2 Dynamic CFAR Threshold](#272-dynamic-cfar-threshold)
  - [2.8 Antenna Configuration Parameters](#28-antenna-configuration-parameters)
    - [2.8.1 Antenna Position Format](#281-antenna-position-format)
  - [2.9 DOA Estimation Parameters](#29-doa-estimation-parameters)
  - [2.10 Region Definition Parameters (SOD)](#210-region-definition-parameters-sod)
    - [2.10.1 Seat Region Definition](#2101-seat-region-definition)
    - [2.10.2 SOD Processing Parameters](#2102-sod-processing-parameters)
    - [2.10.3 Parameter Details](#2103-parameter-details)
- [Chapter 3: Core Algorithm Implementation](#chapter-3-core-algorithm-implementation)
  - [3.1 Algorithm Class Structure](#31-algorithm-class-structure)
  - [3.2 Data Flow Overview](#32-data-flow-overview)
  - [3.3 CIR Combination](#33-cir-combination)
    - [3.3.1 Purpose](#331-purpose)
    - [3.3.2 Configuration](#332-configuration)
    - [3.3.3 Mathematical Description](#333-mathematical-description)
  - [3.4 Frame FFT (2D-FFT)](#34-frame-fft-2d-fft)
    - [3.4.1 Purpose](#341-purpose)
    - [3.4.2 Configuration](#342-configuration)
    - [3.4.3 Mathematical Description](#343-mathematical-description)
  - [3.5 SISO Channel Combination](#35-siso-channel-combination)
    - [3.5.1 Purpose](#351-purpose)
    - [3.5.2 Configuration](#352-configuration)
    - [3.5.3 Mathematical Description](#353-mathematical-description)
  - [3.6 Noise Variance Estimation (NVE)](#36-noise-variance-estimation-nve)
    - [3.6.1 Purpose](#361-purpose)
  - [3.7 CFAR Detection](#37-cfar-detection)
    - [3.7.1 Purpose](#371-purpose)
    - [3.7.2 Configuration](#372-configuration)
    - [3.7.3 Processing Flow](#373-processing-flow)
    - [3.7.4 Sub-bin Refinement](#374-sub-bin-refinement)
    - [3.7.5 Dynamic Threshold Post-filter](#375-dynamic-threshold-post-filter)
  - [3.8 DOA Estimation](#38-doa-estimation)
    - [3.8.1 Purpose](#381-purpose)
    - [3.8.2 Configuration](#382-configuration)
    - [3.8.3 Steering Vector Pre-computation](#383-steering-vector-pre-computation)
    - [3.8.4 Azimuth DBF](#384-azimuth-dbf)
    - [3.8.5 Elevation Phase Comparison Method](#385-elevation-phase-comparison-method)
- [Chapter 4: Socket Server Usage](#chapter-4-socket-server-usage)
  - [4.1 Overview](#41-overview)
  - [4.2 Architecture](#42-architecture)
  - [4.3 Command Line Arguments](#43-command-line-arguments)
  - [4.4 Response Data Structure](#44-response-data-structure)
  - [4.5 Running the Server](#45-running-the-server)
    - [4.5.1 Development Mode](#451-development-mode)
    - [4.5.2 Production Mode (Executable)](#452-production-mode-executable)
  - [4.6 Integration with GUI](#46-integration-with-gui)
  - [4.7 Troubleshooting](#47-troubleshooting)
    - [4.7.1 Common Issues](#471-common-issues)
- [Chapter 5: Batch Processing Guide](#chapter-5-batch-processing-guide)
  - [5.1 Overview](#51-overview)
  - [5.2 Architecture](#52-architecture)
  - [5.3 Configuration](#53-configuration)
    - [5.3.1 Main Parameters](#531-main-parameters)
    - [5.3.2 Path Structure](#532-path-structure)
  - [5.4 Command Line Interface](#54-command-line-interface)
    - [5.4.1 Argument Details](#541-argument-details)
  - [5.5 Output Files](#55-output-files)
  - [5.6 Running Batch Processing](#56-running-batch-processing)
    - [5.6.1 Basic Usage](#561-basic-usage)
    - [5.6.2 Customizing for Different Projects](#562-customizing-for-different-projects)
- [Glossary](#glossary)

## List of Figures (图表目录)

*(No figures listed in the document / 文档内无图表插图)*

## List of Tables (表格清单)

- Table 2: CPD Algorithm Processing Pipeline (CPD 算法处理流水线)
- Table 3: General Control Parameters (通用控制参数)
- Table 4: Waveform Parameters (波形参数)
- Table 5: Signal Processing Parameters (信号处理参数)
- Table 6: Doppler Processing Parameters (多普勒处理参数)
- Table 7: Channel Combination Parameters (通道合并参数)
- Table 8: CFAR Detection Parameters (CFAR 检测参数)
- Table 9: Antenna Configuration Parameters (天线配置参数)
- Table 10: DOA Estimation Parameters (DOA 估算参数)
- Table 11: SOD Region Parameters (SOD 区域参数)
- Table 12: Socket Server Network Configuration (Socket 服务器网络配置)
- Table 13: Socket Server Command Line Arguments (Socket 服务器命令行参数)
- Table 14: Region Result Array Format (区域结果数组格式)
- Table 15: Batch Processing Arguments (批处理参数)
- Table 16: Batch Processing Output Files (批处理输出文件)

## List of Abbreviations (缩略语清单)

| Abbreviation | Full Name                      | 中文释义          |
| ------------ | ------------------------------ | ----------------- |
| **CIR**      | Channel Impulse Response       | 通道冲激响应      |
| **CPD**      | Child Presence Detection       | 儿童遗留检测      |
| **FFT**      | Fast Fourier Transform         | 快速傅里叶变换    |
| **CFAR**     | Constant False Alarm Rate      | 恒虚警率检测      |
| **DOA**      | Direction of Arrival           | 波达方向估算      |
| **DBF**      | Digital Beamforming            | 数字波束成形      |
| **SOD**      | Seat Occupancy Detection       | 座椅占用检测      |
| **ML**       | Machine Learning               | 机器学习          |
| **NVE**      | Noise Variance Estimation      | 噪声方差估计      |
| **SISO**     | Single Input Single Output     | 单输入单输出      |
| **MIMO**     | Multiple Input Multiple Output | 多输入多输出      |
| **SNR**      | Signal-to-Noise Ratio          | 信噪比            |
| **RX**       | Receiver                       | 接收机 (接收通道) |
| **TX**       | Transmitter                    | 发射机 (发射通道) |

## List of Symbols (符号清单)

- $\theta$ : Azimuth angle (方位角)
- $\phi$ : Elevation angle (俯仰角)
- $R$ : Range (距离)
- $V$ : Velocity / Doppler (速度/多普勒)

# Chapter 1: Introduction (引言)

### 1.1 Overview (概述)

本手册为部署于 2T4R (2发射，4接收) 超宽带 (UWB) 雷达系统中的 Dubhe CPD (儿童遗留检测) 算法提供全面的使用和配置指南。该算法通过处理通道冲激响应 (CIR) 数据来高精度检测车内座椅上的乘员状态，并利用机器学习与空间定位技术区分不同的乘员类型 (成人、儿童或空座)。

### 1.2 System Architecture (系统架构)

CPD 算法系统主要由以下核心组件构成：

1. **信号处理核心 (`uwb_algo_cpd_M0_UAB.py`)**
   - 实现雷达信号处理流水线的核心 Python 模块。
   - 包含 2D-FFT 变换、CFAR 目标候选检测、非相干信道合并、DOA 角度估计以及 ML 分类等完整逻辑。
   - 完全支持通过外部 YAML 配置文件进行快速参数修改与定制。
2. **Socket 服务器 (`uwb_algo_socket_server_cpd.py`)**
   - 实时数据流处理服务器。
   - 通过 UDP Socket 接收来自于底层硬件传输的 CIR 原始数据，并将计算出的探测、分类状态推送给上位机 GUI。
   - 支持本地开发模式脚本运行以及用于部署环境的独立免安装可执行程序 (exe) 运行。
3. **离线批处理工具 (`batch_run_RSP.py`)**
   - 用于离线大批量雷达路测数据的自动化回归测试。
   - 支持多进程并行调度执行，极大提升回归迭代效率。
4. **配置文件集 (`config/\*.yaml`)**
   - 定义了信号处理参数、雷达天线布局、探测空间边界以及座舱空间映射区域。

### 1.3 Key Features (核心特性)

- **硬核实时响应**：支持与真实物理雷达硬件的低延迟高速数据流联调。
- **高精度融合分类**：将经典雷达空间谱信号处理算法与轻量级神经网络分类分类模型（Machine Learning Classification）相结合。
- **高度可配置化**：通过 YAML 进行软硬件接口解耦，易于匹配不同的座舱结构与雷达天线。
- **无依赖快捷部署**：可通过内置打包逻辑将复杂的 Python 科学计算栈打包为免运行环境部署的可执行二进制程序。
- **全方位 Debug 支持**：自带详细的数据落盘（Dump）机制和中间节点谱图的可视化绘图功能。

### 1.4 Algorithm Pipeline (算法流水线)

CPD 算法按照多阶段数据流水线执行：

**Table 2: CPD Algorithm Processing Pipeline**

| Stage                   | Function                                                     | Output                                               |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| **CIR Combine**         | 累加多帧 CIR 进行相干积累，抑制非相干噪声                    | 积累增强后的一维 CIR 矢量                            |
| **Phase Compensation**  | 补偿多路射频通道的不平衡度、泄漏（leakage）以及初始相位偏置  | 零范围对齐、相位校正后的 CIR                         |
| **Bandwidth Extension** | （可选阶段）通过 Burg 算法或超分辨率外推拓展射频带宽         | 高分辨率细分 CIR                                     |
| **Frame FFT**           | 沿慢时间维度进行一维 FFT 变换以构建 Range-Doppler 空间       | 二维 Range-Doppler 能量图                            |
| **SISO Combine**        | 跨所选的虚拟天线通道进行非相干能量叠加                       | 合并后的二维 RD 图                                   |
| **CFAR Detection**      | 在二维 RD 图上进行动态恒虚警率检测（基于 NVE 噪声估计）      | 目标单元候选集掩膜                                   |
| **DOA Estimation**      | 基于数字波束成形 (Azimuth DBF) 与俯仰通道相位差估算物理坐标  | 目标三维空间点云坐标 ($R, \theta, \phi, \text{SNR}$) |
| **ML Classification**   | 将选定周期内的特征矩阵送入轻量神经网络                       | 乘员行为及分类标识 (Occupancy Flags)                 |
| **SOD**                 | 映射到定义的座椅几何包围盒进行 DBSCAN 聚类平滑滤波并输出状态 | 5 席座椅独立占用状态指示结果                         |

### 1.5 File Structure (工程目录结构)

项目文件夹默认组织架构如下：

```
2T4R_CPD_DEMO/
├── config/                         # 外部参数配置文件目录
│   ├── dubhe_config.yaml           # 离线批处理与通用配置
│   ├── dubhe_config_online.yaml    # 实时网络调试配置
│   └── ...
├── Data/                           # 模型权重与输入/输出本地媒介
│   ├── model_best.h5              # 训练就绪的神经网络分类模型
│   └── XXXX.h5                     # 针对客户定制底座优化后的特定模型
├── uwb_algo_cpd_M0_UAB.py          # 雷达信号处理与物理算法核心模块
├── uwb_algo_socket_server_cpd.py   # Socket 通信与实时处理框架
├── batch_run_RSP.py                # 多进程离线批处理回归测试脚本
├── utils.py                        # 矩阵运算与坐标投影等底层工具箱
└── dist/
    └── uwb_algo_socket_server_cpd.exe  # PyInstaller 编译生成的 standalone 部署程序
```

### 1.6 Hardware Requirements (硬件环境需求)

- **雷达主片**：Dubhe M0 UWB 雷达核心套件 (2T4R 物理及虚拟天线阵列架构)。
- **天线排布**：当前软件流水线仅原生适配 L 型（L-shape）及梯形（trapezoidal）天线射频极板阵列。
- **宿主系统**：推荐采用 x86_64 Windows 或定制嵌入式 Linux 运算平台。
- **内存开销**：由于维护多路历史高频 CIR 环形缓冲区（Ring Buffer），实时运行推荐空闲 RAM $\ge 8\text{GB}$。

### 1.7 Software Dependencies (软件依赖要求)

运行开发环境需配置以下标准的 Python 环境组件：

```
python == 3.8.17
numpy >= 1.21.0
scipy >= 1.7.0
pandas >= 1.3.0
pyyaml >= 5.4.0
tensorflow >= 2.8.0
h5py >= 3.6.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
```

# Chapter 2: Configuration File Guide (配置文件指南)

算法使用 YAML 文件统一进行参数生命周期管理。本章将详细拆解每个参数的具体数值边界与控制行为。

### 2.1 Overview (概述)

所有 `.yaml` 文件皆存放于 `./config/` 下，核心配置文件包含：

- `dubhe_T_ver4_a.yaml` - 某乘用车实车项目座舱定制参数文件。
- `dubhe_config.yaml` - 通用离线回归批处理基础参数文件。
- `dubhe_config_online.yaml` - 实时 UDP Socket 数据流调试专用参数文件。

### 2.2 General Control Parameters (通用控制参数)

用于调度整个雷达数据流以及数据落盘状态：

**Table 3: General Control Parameters**

| Parameter        | Type | Typical Value | Description                                                  |
| ---------------- | ---- | ------------- | ------------------------------------------------------------ |
| `single_cir_num` | int  | 100           | 单个二进制 `.bin` 数据流文件中包含的 CIR 数据总帧数          |
| `plt_mode`       | int  | 0             | 绘图调试模式：0=不绘图，1=仅绘制结果，2=精细绘制 RD 图，3=全阶段 Debug 绘图 |
| `dump`           | int  | 0             | 运行中间矩阵落盘开关：0=关闭，1=开启（输出 `.mat` 或 `.npz`） |
| `batch_run_mode` | int  | 1             | 批处理/实时选择模式：0=实时模式（配合 Socket 接收器），1=离线批处理模式 |
| `int_cir_step`   | int  | 1             | CIR 数据帧的计算步进跨度                                     |

#### 2.2.1 Parameter Details

- **`single_cir_num`**
  - **取值范围**：1 - 1000
  - **作用机制**：雷达物理采集板在写入落盘文件时，为了避免单文件过大，会将每 100 帧（标准 2T4R 帧率）CIR 作为一个二进制文件切片。此参数告知解析器读取单个 `.bin` 文件时需要循环偏移的帧计数值。
- **`plt_mode`**
  - **取值范围**：0, 1, 2, 3
  - **核心差异**：
    - 设置为 `0` 时运行效率最高。如果回归测试的主要目的仅是获取点云数据（导出 `tgt_info_*.csv`）和占用结果，必须将其设为 `0`。
    - 设置为 `3` 时会在各个数据流转接点触发交互式绘图展示（如 NVE 噪声基线、1D-DBF 谱线等），运行效率将大幅下降，仅建议用于现场单文件深度对齐。

### 2.3 Waveform Parameters (雷达波形参数)

此部分参数主要用于辅助信号处理流水线在数学变换中的距离和多普勒解算：

**Table 4: Waveform Parameters**

| Parameter | Type  | Typical Value | Description                                                  |
| --------- | ----- | ------------- | ------------------------------------------------------------ |
| `num_rx`  | int   | 8             | 系统逻辑/虚拟接收通道数（发射 TX 乘以物理接收极板物理通道数） |
| `cir_len` | int   | 32            | 单路天线输出的 CIR 原始复数采样点长度                        |
| `period`  | float | 0.018         | 单帧雷达帧发射脉冲重复周期（秒），18ms                       |
| `res_rng` | float | 0.15          | 物理波形解析对应的距离分辨率（米/BIN），由射频扫频带宽决定   |

*注意：上述参数由物理射频前端硬件与 SDK 固件确定，此处的参数配置仅作为 Python 端解译点云绝对空间距离与物理速度的输入因子，修改此处的数值并不会反向影响底层雷达硬件的扫频宽度或调制脉冲周期。*

#### 2.3.1 Parameter Details

- **`num_rx`**

  - 对于标准的 2T4R 系统，通过 TDM（时分复用）MIMO 技术，合成后的虚拟通道数一般为 $2 \times 4 = 8$。

- **`period`**

  - 规定物理帧间的时间跨度，用以直接决定最大不模糊速度上限 $v_{\max}$：

    $$v_{\max} = \frac{\lambda}{4 \times T}$$

    其中 $\lambda$ 为雷达的工作电磁波波长。

### 2.4 Raw CIR Processing Parameters (一维 CIR 处理参数)

**Table 5: Signal Processing Parameters**

| Parameter         | Type | Typical Value | Description                                              |
| ----------------- | ---- | ------------- | -------------------------------------------------------- |
| `cir_combine_num` | int  | 9             | 单次相干积累的一维复数 CIR 数据帧数                      |
| `slide_step`      | int  | 27            | 数据环形缓冲区滑动更新步长                               |
| `ring_buffer_len` | int  | 60            | 用于 2D-FFT 慢时间频谱运算的环形缓冲区长度（代表帧深度） |
| `snr_boost_en`    | int  | 1             | 一维背景降噪/SNR 提升算法开关：0=不使能，1=使能          |

#### 2.4.1 CIR Combination and Sliding Window (累加与滑动窗口关系)

这些底层的数学步进参数共同决定了在慢时间维度的覆盖范围：

```
# 每次 2D-FFT 慢时间轴所对应的一维 CIR 总帧数（即慢时间观测时长范围）
total_cir_num = ring_buffer_len * cir_combine_num
```

在标准默认配置下：

- `total_cir_num` = $60 \times 9 = 540$ 帧一维 CIR。
- `slide_step` = 27 帧一维 CIR。

由于雷达数据流是连续灌入的，`slide_step` 必须是 `cir_combine_num` 的整数倍。这意味着慢时间维度上的 FFT 滑动窗口是在相干积累帧的步进基础之上进行的。

#### 2.4.2 SNR Boost

- 当 `snr_boost_en` 打开时，会在频域计算之前对一维复 CIR 执行针对静态泄露自适应估计与减除的降噪算法，平均能为有用呼吸信号提高约 10 dBm 左右的信噪比底限。

### 2.5 Doppler Processing Parameters (多普勒处理参数)

**Table 6: Doppler Processing Parameters**

| Parameter            | Type | Typical Value | Description                                             |
| -------------------- | ---- | ------------- | ------------------------------------------------------- |
| `doppler_win_en`     | int  | 1             | 慢时间加窗开关：0=矩形窗，1= Chebyshev（切比雪夫）窗    |
| `doppler_win_coef`   | int  | 60            | Chebyshev 窗的副瓣抑制衰减分贝数（dB）                  |
| `doppler_dc_en`      | int  | 1             | 慢时间维度静态 DC 分量减除开关：0=保留静态分量，1=减除  |
| `doppler_fft`        | int  | 64            | 多普勒方向进行一维 FFT 变换时的点数大小                 |
| `doppler_fft_scaler` | int  | 64            | FFT 逆变换/正变换后的振幅自适应缩放因子                 |
| `leakage_offset_en`  | int  | 1             | 原始直通泄露（Leakage）手动对齐偏置开关：0=关闭，1=开启 |
| `leakage_offset`     | int  | 5             | 雷达物理前突直通泄露信号所占据的一维 Range Bin 数量值   |

#### 2.5.1 Leakage Offset (直通泄露抑制对齐)

在超宽带雷达近距离探测中，由于天线收发极板存在不可避免的直通耦合泄露（Leakage），会导致 Range 轴起始的前几个 Bin 能量处于极高饱和状态。 为了将物理零点距离映射到数组索引零位，需要通过 `leakage_offset` 对 Range 轴上的数据进行循环位移操作（np.roll），把前段的高饱和泄露移位至数组尾部，保证车内空间第一个物理采样值对应数组的首位：

```
# 在 frame_fft() 内部的处理逻辑
if proc_cfg.lekeage_offset_manaul_en:
    for rx in range(proc_cfg.num_rx):
        fft2d[:,:,:,rx] = np.roll(
            fft2d[:,:,:,rx],
            -proc_cfg.leakage_offset,
            axis=-1
        )
```

### 2.6 Channel Combination Parameters (信道非相干合并参数)

**Table 7: Channel Combination Parameters**

| Parameter | Type | Typical Value   | Description                                |
| --------- | ---- | --------------- | ------------------------------------------ |
| `siso_ch` | list | [0,1,2,3,4,5,6] | 参与 SISO 非相干叠加的虚拟通道通道索引列表 |

虚拟接收通道合并策略取决于真实的物理天线走线及反射隔离程度：

- 当射频极板走线开启 Same-pin RLS（同源抑制）时，推荐选择漏电信号较弱且增益一致的一组通道，如 `[0, 2, 5, 7]`。
- 若硬件环境整体直通干扰极弱且所有通道增益标定优秀，推荐配置全虚拟通道合并：`[0, 1, 2, 3, 4, 5, 6, 7]`。

### 2.7 CFAR Detection Parameters (恒虚警率检测参数)

**Table 8: CFAR Detection Parameters**

| Parameter                | Type | Typical Value  | Description                                                  |
| ------------------------ | ---- | -------------- | ------------------------------------------------------------ |
| `cfar_range_peak_flag`   | int  | 0              | 候选目标点筛选是否要求其必须在 Range 轴上为局部极大值点      |
| `cfar_doppler_peak_flag` | int  | 1              | 候选目标点筛选是否要求其必须在 Doppler 轴上为局部极大值点    |
| `cfar_low_r_idx`         | int  | 2              | CFAR 探测启动的 Range 最小搜索距离门限索引                   |
| `cfar_high_r_idx`        | int  | 20             | CFAR 探测截止的 Range 最大搜索距离门限索引                   |
| `cfar_low_v_idx`         | int  | 1              | 半多普勒对称区域内的最小多普勒索引，用于抑制静态 clutter 分量 |
| `cfar_high_v_idx`        | int  | 20             | 呼吸目标所能触达的最大合理慢多普勒速度解算索引界限           |
| `cfar_th`                | list | [7,7]          | 基础静态 CFAR 门限（dB）。当前双阈值简化架构中仅第一维数值生效 |
| `noi_edges`              | list | [-15,-5,10,20] | 基于 NVE 的自适应背景环境噪声划分边界（分贝）                |
| `cfar_th_dynamic`        | list | [15,12,10,8,7] | 针对不同自适应环境噪声区间分别采用的动态门限（分贝）         |

#### 2.7.1 Doppler Velocity Constraints (多普勒速度门限过滤)

为了最大化滤除汽车内饰震动或缓慢温漂引起的近零频干扰（DC 泄露残余），并通过截止频率限制非相关的高频运动目标（如车外大范围动态行人），算法在 CFAR 检测中引入了对称的双侧速度通带滤波器门限：

- **`cfar_low_v_idx`**：设定 DC 遮罩，将慢多普勒维度的接近零频的部分剔除：

  - 多普勒维度的 $[0, cfar\_low\_v\_idx)$ 以及 $(doppler\_fft - cfar\_low\_v\_idx, doppler\_fft]$ 区域被全部强制遮罩。

- **`cfar_high_v_idx`**：设置运动上限，将慢多普勒极速变化区域排除。 最终在 2D-FFT Range-Doppler 图上，仅保留两半对称扇形频谱通带中的有效目标：

  $$\text{Doppler Valid Range} = [cfar\_low\_v\_idx, cfar\_high\_v\_idx) \cup (doppler\_fft - cfar\_high\_v\_idx, doppler\_fft - cfar\_low\_v\_idx]$$

#### 2.7.2 Dynamic CFAR Threshold (双阶段自适应动态门限)

由于车辆座舱环境在静止、空调开启或高速路试（引擎与底盘震动注入）等不同工况下的噪声本底相差极大，静态单一阈值易造成虚警。 算法依据当前的 NVE 评估对雷达噪声本底执行 5 区段分区标定：

- `noi_edges` 设定了 4 个划分界限：$[-15, -5, 10, 20]\text{ dB}$。
- 其将外部环境本底噪声分为 5 个阶梯。每个阶梯自适应触发对应的 `cfar_th_dynamic` 阈值：
  - 本底极低区间 $(-\infty, -15]\text{ dB}$：使用极保守阈值 $15\text{ dB}$（保障极致信噪比纯净度）。
  - 低噪区间 $(-15, -5]\text{ dB}$：使用自适应门限 $12\text{ dB}$。
  - 中噪本底区间 $(-5, 10]\text{ dB}$：使用基准平衡门限 $10\text{ dB}$。
  - 强震动噪区间 $(10, 20]\text{ dB}$：降低系统敏感度门限至 $8\text{ dB}$。
  - 极高恶劣噪区间 $(20, +\infty)\text{ dB}$：使用极高阈值 $7\text{ dB}$（通过降低门限优先防止目标淹没，并引入后续 DBF 谱质筛选二次校验）。

### 2.8 Antenna Configuration Parameters (天线位置及参数标定)

**Table 9: Antenna Configuration Parameters**

| Parameter            | Type | Typical Value | Description                                                  |
| -------------------- | ---- | ------------- | ------------------------------------------------------------ |
| `ant_calib_en`       | int  | 1             | 射频阵列初相标定开关：0=关闭，1=基于标定数组自适应校正相位   |
| `ant_calib_phase`    | list | 8 floats      | 用于方位角（Azimuth）解算的 8 通道复数初相补偿矩阵（弧度值） |
| `ant_calib_phase_y`  | list | 8 floats      | 用于俯仰角（Elevation）计算的 8 通道复数初相补偿矩阵（弧度值） |
| `ant_pos`            | list | 8x2 array     | 8 个物理与虚拟阵元在二维网格坐标系上的投影归一化坐标值 ($[x, y]$，单位：波长 $\lambda$) |
| `ant_dbf1d_select`   | list | [2,3,6,7]     | 参与 1D 数字波束成形（方位角计算）的天线虚拟通道索引         |
| `ant_el_select`      | list | [7,5]         | 用于俯仰相位差比较法的一对物理垂直通道索引                   |
| `ant_multi_path_sel` | list | [2,6]         | 参与多径反射异常强度判定的射频参考通道                       |

#### 2.8.1 Antenna Position Format (雷达波长物理排布格式)

天线相对坐标的输入形式为 $[x, y]$，单位为信号波长。这在很大程度上决定了空间谱数字波束成形（Digital Beamforming）的相位引导矢量构造。 以下是典型的 L 型虚拟天线坐标（以 Channel 0 作为一维与二维极化物理零点参考点）：

```
ant_pos: [
    [0.0, 0.0],    # Channel 0: Reference Origin
    [1.0, 0.0],    # Channel 1: x轴向偏移 1.0 个电磁扫频波长
    [0.5, -0.5],   # Channel 2: x轴偏 0.5λ, y轴负向偏 0.5λ
    [1.0, -0.5],   # Channel 3: ...
    [1.0, 0.0],    # Channel 4
    [2.0, 0.0],    # Channel 5
    [1.5, -0.5],   # Channel 6
    [2.0, -0.5]    # Channel 7
]
```

### 2.9 DOA Estimation Parameters (波达方向解算参数)

**Table 10: DOA Estimation Parameters**

| Parameter         | Type  | Typical Value | Description                                                  |
| ----------------- | ----- | ------------- | ------------------------------------------------------------ |
| `azi_angle_range` | list  | [-70, 70]     | 方位角数字扫描的起始角与终止角范围（角度，度值）             |
| `ele_angle_range` | list  | [-60, 60]     | 俯仰角自适应解算的边界保护阈值（角度，度值）                 |
| `azimuth_num`     | int   | 64            | 空间角扫描格点网格细分数（用于划分 1D-DBF 的波束密集程度）   |
| `elevation_num`   | int   | 16            | 俯仰空间扫频格点数（在当前相位直接对比算法版本下属于无效保留参数） |
| `sidelobe_th`     | float | 0             | 角度谱多峰识别中的主瓣/旁瓣抑制抑制能量比（dB）              |
| `multi_path_rm`   | float | 40            | 多径折射引起的镜像假目标判定角门限（度值）                   |
| `dbf_diff`        | float | 8             | DBF 扫描谱主瓣峰度滤波门限（低于此值的点被视作非相关噪声，强制移除） |
| `mount_el_angle`  | float | -20           | 雷达实车物理倾斜安装角补偿值（用于纠正旋转后的绝对坐标系偏置） |

### 2.10 Region Definition Parameters (座椅空间映射与 SOD 映射参数)

此部分主要用于定义目标空间点云归属于车内哪个物理座椅，从而实现座椅占用检测 (SOD)。

#### 2.10.1 Seat Region Definition (车座 3D 几何映射范围)

每个座椅被定义为在方位角、俯仰角、距离三维极坐标系下的空间几何三维包围盒（3D bounding box）：

```
[[azimuth_min, azimuth_max],      # 空间方位扫描角度通带范围 (degrees)
 [elevation_min, elevation_max],  # 空间俯仰扫描角度通带范围 (degrees)
 [range_min, range_max]]           # 距离段阈值门限 (当前算法逻辑不使能，以高度极角解算为准)
```

**ET5 实车物理座舱配置标定范例：**

```
SEAT_A: [  # 驾驶席 (Driver seat - 左前)
    [-50, -15],     # 方位角范围：-50° 至 -15°
    [ -5,  40],     # 俯仰角范围：-5° 至 40°
    [-0.3, 1.0]]    # ignore
SEAT_B: [  # 副驾驶席 (Passenger seat - 右前)
    [ 15,  50],     # 方位角范围：15° 至 50°
    [ -5,  40],     # 俯仰角范围：-5° 至 40°
    [-0.3, 1.0]]    # ignore
SEAT_C: [  # 后排右侧席 (Rear Right)
    [ 10,  50],     # 后排在雷达俯仰看下去呈大夹角：10° 至 50°
    [-60,   0],     # 仰角物理位置倾斜向下：-60° 至 0°
    [-1.0, 0.0]]    # ignore
SEAT_D: [  # 后排中间席 (Rear Middle)
    [-10,  10],     # 居中对称夹角：-10° 至 10°
    [-60,   0],     # 俯角深陷：-60° 至 0°
    [-1.0, 0.0]]    # ignore
SEAT_E: [  # 后排左侧席 (Rear Left)
    [-50, -10],     # 后排左方位包围：-50° 至 -10°
    [-60,   0],     # 俯角：-60° 至 0°
    [-1.0, 0.0]]    # ignore
```

#### 2.10.2 SOD Processing Parameters (座标决策层参数)

**Table 11: SOD Region Parameters**

| Parameter     | Type | Typical Value | Description                                            |
| ------------- | ---- | ------------- | ------------------------------------------------------ |
| `REGION_NUM`  | int  | 5             | 参与系统决策的车内有效座席区域总量（SEAT_A 到 SEAT_E） |
| `REGION_BUF`  | int  | 3             | DBSCAN 时间平滑多帧联合累积历史记录缓冲窗帧数          |
| `MAX_RNG_IDX` | int  | 12            | 过滤外部干扰（如车外大目标）的最大距离索引边界限制     |

#### 2.10.3 Parameter Details

- **`REGION_BUF`**
  - **物理意义**：算法层采用 DBSCAN 聚类算法在极空间进行区域判别时，若仅单帧输出，点云极易在呼吸临界或车身晃动时发生闪烁。算法在内存中维护了一个长度为 `REGION_BUF` 的三维时空滑窗。数值越大，多帧时空轨迹聚类输出越稳定，但会带来约数帧的决策输出延迟（Latency）。
- **`MAX_RNG_IDX`**
  - **物理意义**：强制丢弃远端多径反射虚警目标。距离由于被乘上距离分辨率（res_rng = 0.15m），因此 $12 \times 0.15\text{ m} = 1.8\text{ m}$ 之外的点云云集将会被完全无视。这可以有效防止非座舱内部的目标误触系统的儿童遗留警报。

# Chapter 3: Core Algorithm Implementation (算法核心原理)

本章详尽剖析核心雷达空间谱信号处理算法类的数学演化与其内部的代码架构和工程设计逻辑。

### 3.1 Algorithm Class Structure (算法底层类框架)

核心逻辑全部封装在 `Algo_CPD` 类中（`uwb_algo_cpd_M0_UAB.py`）：

```
class Algo_CPD:
    def __init__(self, args, CIRGranularity, Max_CIRNum, 
                 CIR_Len, RxNum, auc_num=0, gen_h5=False, 
                 proc_cfg_in=None):
        # 1. 初始化雷达环形 FIFO 缓冲区、慢时间维度的快慢时矩阵等底层存储器结构
        # 2. 预先解析并读入自适应神经网络模型权重权重（Keras/TF/XXXX.h5）
        # 3. 在极角空间下根据 ant_pos 与 ant_dbf1d_select 预先计算出 1D-DBF 的空载相位引导方向矩阵 (Steering Vectors)
        pass
        
    def run(self, cir):
        # 系统物理实时输入主调度接口
        # 实时接收外界单帧虚拟 8 通道一维复数 CIR 数据，并将其推入内部双端快慢时循环累积队列。
        # 当慢时间轴 FIFO 缓冲区填充深度达到 ring_buffer_len 时，开始提取对应矩阵块并激活 cpd_core 计算机制
        pass
        
    def cpd_core(self, cirs, cnt):
        # 雷达信号处理、空间三维极坐标映射、多帧自适应检测等完整功能流
        # 返回目标点云和座舱占用判决标识 (Region_result)
        pass
```

### 3.2 Data Flow Overview (物理时钟与计算数据流)

座舱儿童遗留检测（CPD/SOD）属于低频高可靠性决策需求，呼吸检测与分类极其考验算法的稳定性。下图说明了时钟、缓冲区设计以及从物理信号到物理占用决策矩阵的转换：

- **数据维数变换**：系统输入规格为 8 通道一维复数 CIR 组成的 $8 \times 32$ 阶快慢时一维快时间切片。单物理帧生成耗时 $18\text{ ms}$。
- **相干相加（CIR Combine）**：每隔 `cir_combine_num = 9` 个雷达帧进行复数 coherent 相干累加，以此提升约 10dB 的一维信噪比。其完成一帧计算对应的积累时间跨度为 $18\text{ ms} \times 9 = 162\text{ ms}$。
- **时空决策轴（2D-FFT 跨度）**：2D-FFT 的慢时间周期深度设定为 `ring_buffer_len = 60`。当慢时间维度填满，系统将处理长达 $60 \times 162\text{ ms} \approx 9.72\text{ s}$ 的相干谱数据切片。
- **重计算更新（FIFO 滚动刷新步长）**：慢时间矩阵不需要等待 $9.72\text{ s}$ 才重新更新计算。数据缓冲区通过 FIFO 队列先进先出管理，滚动滑动步进设定为 `slide_step = 27`，意味着当在时间维度上累积满 27 帧一维 CIR（耗时 $27 \times 18\text{ ms} = 486\text{ ms}$，即约 $0.5\text{ s}$）时，就会重新执行一轮计算，刷新上位机输出点云与决策信息。
- **计算互锁保护（二级缓存）**：为确保离线与实时 Socket 在重构 2D-FFT 频谱矩阵时，耗时较长的三维空间定位（1D-DBF + 呼吸神经网络分类推理）计算不因高频物理串口数据传入发生内存覆盖，在 `run` 的接口内部设计了二级临时缓冲区，负责在计算期间暂存并排队外部物理帧。

数据流物理演进过程如下：

```
【雷达芯片物理极板】
  └─ Rx1~4 物理通道, Tx1~2 时分复用 (TDM) MIMO 扫频输出
         │ 
         ▼ [物理硬件时钟：18 ms / Frame]
【1帧原始复数通道矢量数据】 $8 \times 32$ 复数数据矩阵
         │
         ▼ [自适应积累：9帧相干相加]
【 coherent 合并一阶矢量】 快时间一阶增强, 抑制偶发性多径相位抖动 (耗时: 162 ms)
         │
         ▼ [数据自适应背景估计]
【一维背景相移校准、直通泄漏减除与 SNR Boost 自适应前处理】 降低底噪约 10dBm
         │
         ▼ [先进先出慢时间累积队列：填充深度至 60 阶]
【快慢时 2D 复矩阵】 形成大小为 $8 \times 60 \times 32$ 的慢时间观测数据体 (全带宽观测时长 9.72s)
         │
         ▼ [2D-FFT (慢时间向进行 64 点加加切比雪夫窗 FFT 运算与静态 DC 去除)]
【Range-Doppler 功率谱空间】 解算得到各通道空间下的多普勒速度与距离网格
         │
         ▼ [通道非相干合并 (SISO 合并)]
【Combined Range-Doppler 二维能量图】
         │
         ▼ [NVE 噪声本底分析 + 自适应分阶恒虚警检测 (CFAR)]
【目标候选点网格掩膜】 锁住具备显著呼吸频率特征的候选距离 bin 和速度 bin
         │
         ▼ [三维极坐标计算 (1D-DBF 方位谱估计 + 垂直俯仰对相位直接对比)]
【空间物理点云】 获得每个检测点的高精度三维坐标矢量集：[$R$, $\theta$, $\phi$, $\text{SNR}$]
         │
         ▼ [自适应呼吸神经网络模型推理 (Feature Matrix -> ML NN)]
【乘员生物呼吸状态与行为预测】 提取分类：成人/儿童/非生物震动/空座
         │
         ▼ [DBSCAN 三维时空轨迹密度聚类 & 5席座椅包围盒（SEAT_A~E）阈值判决映射]
【座椅状态输出决策】 (Region_result 标志) -> [0=空, 1=成人, 2=儿童] 自适应上报系统总线
```

### 3.3 CIR Combination (通道相干积累)

#### 3.3.1 Purpose

用于在初始一维测距上，通过对慢时间维度上相邻复数帧执行矢量对齐相加，压制白噪声本底，提升检测空间极限。

#### 3.3.2 Configuration

- `cir_combine_num`: 积累帧数（默认：9帧）。

#### 3.3.3 Mathematical Description

给定 $N$ 个相邻物理周期一维 CIR 复数响应矢量 $x_n(t)$，进行相干求和合并输出 $x_{\text{comb}}(t)$：

$$x_{\text{comb}}(t) = \frac{1}{N} \sum_{n=0}^{N-1} x_n(t)$$

在复数域进行相干积累时，有用信号振幅与 $N$ 成正比，而随机高斯白噪声由于相位随机，积累后噪声功率仅与 $N$ 的一次方成正比。由此可获得接近以下公式的理论信噪比改善：

$$\text{SNR Gain (dB)} \approx 10 \log_{10}(N) \quad (\text{当 } N=9 \text{ 时, 理论增益约为 } 9.54\text{ dB})$$

### 3.4 Frame FFT (多普勒解谱 2D-FFT)

#### 3.4.1 Purpose

沿慢时间维度（帧间时间变化）进行快速傅里叶变换，将具有微弱变化规律的座舱乘员呼吸运动特征与静态内饰杂波（Clutter）在频域中分离开来。

#### 3.4.2 Configuration

- `doppler_win_en`: Chebyshev 窗加权使能。
- `doppler_win_coef`: 切比雪夫窗副瓣衰减参数（默认 60dB）。
- `doppler_dc_en`: 静态杂波直流滤除开关。
- `doppler_fft`: 频域离散计算点数（默认 64 点）。

#### 3.4.3 Mathematical Description

在对慢时间维度数据进行 FFT 变换前，为去除静态杂波和直流泄露，需沿慢时间轴减去均值：

$$x_{\text{dc}}(n, r) = x(n, r) - \frac{1}{N} \sum_{n=0}^{N-1} x(n, r)$$

其中 $n \in [0, N-1]$ 代表慢时间样本点（深度 $N = \text{ring\_buffer\_len}$），$r$ 表示一维距离 Bin。 然后，将直流修正后的慢时间信号与窗函数 $w(n)$ 相乘，并计算一维傅里叶变换：

$$X(f_d, r) = \sum_{n=0}^{N-1} x_{\text{dc}}(n, r) \cdot w(n) \cdot e^{-j 2\pi \frac{f_d \cdot n}{N_{\text{fft}}}}$$

其中 $f_d$ 表示多普勒离散频点索引，$N_{\text{fft}}$ 为离散变换点数（即 `doppler_fft` = 64），$w(n)$ 为设计的 Chebyshev 窗系数。

### 3.5 SISO Channel Combination (非相干通道合并)

#### 3.5.1 Purpose

为了融合不同极化物理位置天线的信息，避免因单路通道被障碍物遮挡、身体局部相位抵消造成的虚警与漏检。

#### 3.5.2 Configuration

- `siso_ch`: 指定参与非相干叠加合并的虚拟接收极板天线通道列表。

#### 3.5.3 Mathematical Description

多通道非相干合并，即对各通道一维 FFT 处理后得到的 Range-Doppler 二维复矩阵取模的平方（计算功率谱），然后在空间维做算术平均：

$$P_{\text{comb}}(f_d, r) = \frac{1}{M} \sum_{m=0}^{M-1} |X_m(f_d, r)|^2$$

其中 $M$ 为 `siso_ch` 包含的虚拟天线数量。非相干叠加仅合并幅值信息，消除了相位随机起伏导致的空间干涉凹陷。

### 3.6 Noise Variance Estimation (NVE 自适应背景估计)

#### 3.6.1 Purpose

在 RD 功率谱二维平面上，沿多普勒轴提取指定中、高频多普勒区域对应的频点，进行中值（Median）提取，以逼真反映当前距离 bin 上的热噪声和高频干扰水平。这有助于生成在各 Range Bin 维度上动态变化的自适应噪底估计曲线。

### 3.7 CFAR Detection (恒虚警率目标候选检测)

#### 3.7.1 Purpose

寻找出可能由乘员微弱呼吸起伏引起的，且幅度显著高于当前局部自适应背景噪声水平的 RD 网格坐标。

#### 3.7.2 Configuration

- `cfar_th`: 初始检测信噪比门限。
- `cfar_low_r_idx` / `cfar_high_r_idx`: 车辆座舱探测极轴有效距离区间。
- `cfar_low_v_idx` / `cfar_high_v_idx`: 呼吸目标多普勒门区。
- `cfar_range_peak_flag` / `cfar_doppler_peak_flag`: 局部峰值滤波标志。

#### 3.7.3 Processing Flow

1. **背景噪声谱图展开**：将 NVE 估计生成的一维自适应距离噪底矢量在多普勒维度进行对称复制，以构造二维背景噪底矩阵 $P_{\text{background}}(f_d, r)$。

2. **构建 SNR 谱图**：计算雷达实测 RD 平面与背景噪底矩阵的分贝差值，从而在二维域上重建信噪比图：

   $$\text{SNR}_{\text{power}}(f_d, r)\text{ [dB]} = 10 \log_{10}\left(\frac{P_{\text{comb}}(f_d, r)}{P_{\text{background}}(f_d, r)}\right)$$

3. **初次筛选**：保留 $\text{SNR}_{\text{power}}(f_d, r) > cfar\_th$ 的单元格。

4. **极大值抑制滤波**：若开启 `cfar_range_peak_flag` / `cfar_doppler_peak_flag`，则目标单元还必须是局部小窗口内的局部极大值，以进一步精简和优化检测到的点云。

5. **门区通带掩膜过滤**：剔除落在 $[0, cfar\_low\_r\_idx)$ 与 $(cfar\_high\_r\_idx, \text{cir\_len}]$ 的点云，以及处于多普勒速度限制通带之外的目标。

#### 3.7.4 Sub-bin Refinement (次级亚分辨率精细对齐)

针对选中的网格，在距离多普勒轴相邻区域内采用二次函数进行局部插值（Quadratic Interpolation），以超越 FFT 网格分辨率的物理限制，获取极小偏移量下的精细点位置（距离亚 Bin 与速度亚 Bin 外推），进而为后续的角度谱空间定位稳定度提供有力保障。

#### 3.7.5 Dynamic Threshold Post-filter (动态自适应二次滤波)

对通过 CFAR 标定的所有候选点云，将对应的环境背景本底噪声级别与配置的 `noi_edges` 边缘进行区段比对，触发二次降噪机制：

- 若在标定极高环境噪声段（如空调风口振动），则强制将点云信噪比阈值过滤门限提高，去除虚警。
- 在安静状态下则自动下放门限，以捕获极低呼吸振幅下的生命体。

### 3.8 DOA Estimation (三维空间定位与波达方向估算)

#### 3.8.1 Purpose

计算通过检测的目标候选网格在座舱三维坐标系中的物理方位角 $\theta$ 和俯仰角 $\phi$。

#### 3.8.2 Configuration

- `ant_dbf1d_select`: 方位角 DBF 通道索引。
- `ant_el_select`: 俯仰直接相位对比通道。
- `azi_angle_range`: 扫描方位跨度限制。
- `azimuth_num`: 方位谱划分格点数。

#### 3.8.3 Steering Vector Pre-computation (空间引导矢量标定)

在类初始化时，为避免重复执行高能耗的复指数矩阵运算，算法会根据方位扫描范围预先离散化生成相位引导矢量矩阵 $a(\theta)$：

```
# 生成在物理极轴方位内的离散角矢网格
self.azi_angle = np.linspace(
    self.proc_cfg.azi_angle_range[0],
    self.proc_cfg.azi_angle_range[1],
    self.proc_cfg.azimuth_num,
    endpoint=True
)

# 针对方位 DBF 天线阵列解算三维引导谱矩阵
self.dbf1d_sv = np.exp(-1j * 2 * np.pi * (
    self.logic_ant_pos_azi[self.proc_cfg.ant_dbf1d_select].reshape(-1, 1) *
    np.sin(self.azi_angle / 180 * np.pi).reshape(1, -1)
))

# 射频初始相位偏置校准补偿
if self.proc_cfg.ant_calib_en:
    self.dbf1d_sv = self.dbf1d_sv * \
        self.ant_phase[self.proc_cfg.ant_dbf1d_select].reshape(-1, 1)
```

#### 3.8.4 Azimuth DBF (方位角数字波束成形)

方位角通过经典的数字波束成形（Digital Beamforming, DBF）求解。

**数学物理模型**：对于配置了均匀间距的天线阵列系统，若某虚拟阵元在方位方向的相对坐标为 $d$（以波长为单位），则对于扫过物理角 $\theta$ 的一维波束其在复数域的理论相位延迟引导矢量为：

$$a(\theta) = \left[e^{-j 2\pi \frac{d_0 \sin(\theta)}{\lambda}}, e^{-j 2\pi \frac{d_1 \sin(\theta)}{\lambda}}, \dots, e^{-j 2\pi \frac{d_{M-1} \sin(\theta)}{\lambda}}\right]^T$$

其中 $M$ 为参与方位角计算的物理虚拟天线通道数量。

**功率谱求和解算**：对于从候选 Range-Doppler 目标位置提取的虚拟天线的复响应向量 $x = [X_0, X_1, \dots, X_{M-1}]^T$，各物理扫描角下的 DBF 波束输出响应功率 $P(\theta)$ 如下：

$$P(\theta) = |a^H(\theta) \cdot x|^2$$

其中 $(\cdot)^H$ 为共轭转置算子。 方位角度解算结果即为方位数字谱图中的极大值处对应的扫描物理角 $\theta_{\text{peak}}$：

$$\theta_{\text{target}} = \arg\max_{\theta} P(\theta)$$

#### 3.8.5 Elevation Phase Comparison Method (俯仰角相位对比解算)

与需要多物理天线、大计算开销方位 DBF 扫描不同，俯仰角度解算考虑到垂直方向天线数量较少，本流水线中采用了经典的高能效相位差解算（Phase Comparison）直接求解。

**物理基础原理**：假设垂直排布的物理极板两阵元其垂直物理间距为 $d_y$。当天线前方俯仰角 $\phi$ 的方向上有信号波束入射时，两接收通道之间会产生物理相位差：

$$\Delta \phi = \frac{2\pi d_y \sin(\phi)}{\lambda}$$

因此，通过直接计算两个俯仰测试通道（`ant_el_select`）信号复数实部与虚部的自相关弧度角差值，可以直接计算俯仰角 $\phi$：

$$\phi = \arcsin\left(\frac{\Delta \phi \cdot \lambda}{2\pi d_y}\right)$$

在实际解算中，此相位差 $\Delta \phi$ 的弧度提取依赖于实车参数标定中 `ant_calib_phase_y` 对多路 RF 射频链路固有相位迟滞差值的标定与减除，标定精度将直接影响俯仰角度空间定位的准确性。

# Chapter 4: Socket Server Usage (实时通信调试套件使用手册)

`uwb_algo_socket_server_cpd.py` 整合了实时雷达 CIR 流入接口，并通过标准局域网协议支持实车、台架与上位机实时调试。

### 4.1 Overview (功能概述)

Socket 服务运行于系统后台，充当雷达物理串口数据转发层与 GUI 渲染应用层之间的中央解算桥梁：

1. **数据中转通道（UDP Port: 55555）**：监听来自于 UWB 雷达芯片驱动板，通过 UDP 循环吐出的标准复数一维 CIR 切片包（Data Stream）。
2. **上位机同步通道（UDP Port: 55556）**：通过 JSON 结构向 `DubheUwbApp` 实时传送计算就绪的检测状态与点云定位。
3. **客户系统通道（UDP Port: 55550）**：向车载域控制器（VIP / CDC）等客户终端直接上报简化后的二进制占用判定状态字数组。

### 4.2 Architecture (物理网络架构)

**Table 12: Socket Server Network Configuration**

| Component           | Target Address    | Functional Description                                       |
| ------------------- | ----------------- | ------------------------------------------------------------ |
| **Server Bind**     | `127.0.0.1:55555` | 监听 UWB 硬件网关广播或文件回放串口传入的物理复数 CIR 数据流 |
| **GUI Output**      | `127.0.0.1:55556` | 将打包的目标空间三维点云坐标、信噪比以及中间频谱，发送至上位机进行可视化显示 |
| **Customer Output** | `127.0.0.1:55550` | 推送简化的整车座椅 5 席位独立占用状态标识数组到客户终端      |

### 4.3 Command Line Arguments (命令行可选初始化标识)

Socket 服务在运行时通过传入标志参数自适应加载系统所需要的权重资源：

```
python uwb_algo_socket_server_cpd.py \
    -p ./config/dubhe_config_online.yaml \
    -s ./Data \
    -n data \
    -w ./Data/model_best.h5 \
    --gui_root "D:\Calterah\DubheUwbApp"
```

**Table 13: Socket Server Command Line Arguments**

| Argument Long | Short | Default Value                     | Functional Specification                              |
| ------------- | ----- | --------------------------------- | ----------------------------------------------------- |
| `--proc_cfg`  | `-p`  | `config/dubhe_config_online.yaml` | 待载入的信号处理控制参数 YAML 配置文件路径            |
| `--save_path` | `-s`  | `./Data`                          | 实时联调时，原始复数 CIR 数据流以及点云结果的落盘目录 |
| `--postfix`   | `-n`  | `data`                            | 落盘文件的后缀标志标识                                |
| `--weight`    | `-w`  | `./Data/model_best.h5`            | 编译加载的轻量化神经网络分类权重路径                  |
| `--gui_root`  | -     | `None`                            | 上位机应用执行路径（用于进程联动）                    |

### 4.4 Response Data Structure (输出数据包格式)

`Region_result` 状态决策数组其向总线、上位机或外设吐出的状态包规范定义如下：

**Table 14: Region Result Array Format**

| Array Index | Target Identifier | Payload State Description Mapping                            |
| ----------- | ----------------- | ------------------------------------------------------------ |
| **0**       | `flag`            | 全车综合占用状态汇总标识：`0` = 空置（Empty）, `1` = 成人遗留（Adult）, `2` = 儿童遗留（Child） |
| **1**       | `Seat A`          | 座椅 A (Driver 左前) 物理标志：`0` = 未占用，`1` = 存在乘员占用 |
| **2**       | `Seat B`          | 座椅 B (Passenger 右前) 物理标志：`0` = 未占用，`1` = 存在乘员占用 |
| **3**       | `Seat C`          | 座椅 C (后排右侧席 Rear Right) 物理标志：`0` = 未占用，`1` = 存在乘员占用 |
| **4**       | `Seat D`          | 座椅 D (后排中间席 Rear Middle) 物理标志：`0` = 未占用，`1` = 存在乘员占用 |
| **5**       | `Seat E`          | 座椅 E (后排左侧席 Rear Left) 物理标志：`0` = 未占用，`1` = 存在乘员占用 |
| **6**       | `Seat F`          | 座椅 F 物理标志：`0` = 未占用，`1` = 存在乘员占用，`-1` 表示未配置此席位（Reserved） |
| **7**       | `Seat G`          | 座椅 G 物理标志：`0` = 未占用，`1` = 存在乘员占用，`-1` 表示未配置此席位（Reserved） |
| **8**       | `Seat H`          | 座椅 H 物理标志：`0` = 未占用，`1` = 存在乘员占用，`-1` 表示未配置此席位（Reserved） |

### 4.5 Running the Server (启动机制)

#### 4.5.1 Development Mode (本地开发模式)

在安装有完整 Anaconda 或原生 Python 计算栈的宿主机开发环境中：

```
# 切换至底层工作路径
cd 2T4R_CPD_DEMO

# 直接拉起 Socket 服务并指定配置文件与权重
python uwb_algo_socket_server_cpd.py \
    -p ./config/dubhe_config_online.yaml \
    -w ./Data/model_best.h5
```

#### 4.5.2 Production Mode ( executable 独立可执行模式编译)

如果需要在未安装任何开发包和 Python 环境的环境中运行，可以通过项目自带的 PyInstaller 工具链将环境依赖和权重资源一并打入单个二进制 EXE 中：

```
# 使用 PyInstaller 构建部署包
pyinstaller --onefile \
    --add-data "config/dubhe_config_online.yaml;config" \
    --add-data "Data/model_best.h5;Data" \
    --hidden-import=tensorflow \
    uwb_algo_socket_server_cpd.py

# 编译结束后, 纯净的 EXE 文件将输出在 ./dist/ 下。直接运行测试：
.\dist\uwb_algo_socket_server_cpd.exe
```

### 4.6 Integration with GUI (上位机联调步骤)

1. 在宿主机上双击或通过终端拉起后台 `uwb_algo_socket_server_cpd.exe`。
2. 双击打开 `DubheUwbApp` 可视化图形客户端应用。
3. 设定其连接至环回本地环回端口（一般软件已默认写死 `127.0.0.1:55555`）。
4. 连接成功后，启动雷达。GUI 将实时同步展示 3D 雷达点云的投影和 5 座座椅占用分类色块。

### 4.7 Troubleshooting (故障排除指南)

#### 4.7.1 Common Issues

- **网络套接字绑定失败 (`Error: Address already in use`)**
  - *产生根源*：通常是因为先前拉起的后台 Socket 进程异常挂起或上位机端口未正常关闭，占用了 `55555` 或 `55556` 物理端口。
  - *解决方案*：在终端执行 `taskkill /f /im uwb_algo_socket_server_cpd.exe`（Windows 环境下），强制终结挂起的残留进程；或等待 60 秒直至 TCP 链路完成物理回收后再拉起。
- **配置文件加载缺失 (`Error: Config file not found`)**
  - *产生根源*：相对路径执行上下文发生错乱。
  - *解决方案*：在启动命令行中使用完整的绝对文件系统路径来覆盖配置文件参数，如 `-p D:\2T4R_CPD_DEMO\config\dubhe_config_online.yaml`。
- **神经网络权重损坏及算子冲突 (`Error: Failed to load model weights`)**
  - *产生根源*：TensorFlow 与 Python 科学计算依赖库的版本发生了不兼容。
  - *解决方案*：确保运行环境完全符合 **Chapter 1.7** 规定的软件依赖，特别是 `tensorflow`、`h5py` 与当前运行 Python 版本（如 3.8.17）的二进制版本兼容性。

# Chapter 5: Batch Processing Guide (大批量离线回归测试指南)

`batch_run_RSP.py` 离线批处理分析框架支持通过并行化运算高效进行实车大量采集数据包（Binary Log）的自动分析与参数验证。

### 5.1 Overview (离线测试概述)

离线大批量测试框架用于在算法参数发生迭代时，对以往录制并落盘的数万物理帧进行自动化回归分析，以获取点云特征并评估虚警/漏检率，从而避免新参数影响存量优秀数据集的稳定性。

### 5.2 Architecture (多进程架构逻辑)

为应对庞大的浮点数矩阵运算，批处理框架基于 Master-Worker 设计模式在宿主计算机上拉起物理核心级的进程池：

```
主进程 Master Process (分配路径 & 数据汇总)
  ├── 工作进程 Worker_1 ──► 自适应加载 YAML ──► 处理 Case_Folder_1 (data_0001~0010.bin) ──► 导出 CSV
  ├── 工作进程 Worker_2 ──► 自适应加载 YAML ──► 处理 Case_Folder_2 (data_0011~0020.bin) ──► 导出 CSV
  ├── 工作进程 Worker_3 ──► 自适应加载 YAML ──► 处理 Case_Folder_3 (data_0021~0030.bin) ──► 导出 CSV
  └── ...
```

每个子进程被独立分配到一个子案例文件夹，多进程并行读取二进制原始 CIR 包并执行物理运算，输出不冲突的结果报表。

### 5.3 Configuration (批处理配置)

#### 5.3.1 Main Parameters (脚本主配置入口)

需要打开 `batch_run_RSP.py` 在脚本头部填入批处理运行的路径设置：

```
# 离线数据结算后, 生成的目标报表与点云 CSV 输出的主目录路径
save_path = r"O:\roadtest\System\Dubhe\Regression_Results_V1.1"

# 包含原始海量车内采集数据的总根目录路径 (内部应包含多批次多案例子文件夹)
root_path = r"O:\roadtest\System\Dubhe\Raw_Recordings"

# 此次回归测试指定调用的座舱信号处理与天线标定参数 yaml 配置文件路径
config_file = "./config/dubhe_config.yaml"

# 分配的物理内核并发计算子线程数 (0-32, 推荐设置为逻辑物理 CPU 核心数的一半左右)
num_processes = 4
```

#### 5.3.2 Path Structure (路测数据存储排布规范)

批处理回归工具会自动遍历 `root_path`。输入路径结构必须符合以下标准的二级分批层次结构：

```
root_path/
├── batch_1/                       # 物理大批次 (如: et5_driver_present)
│   ├── case1/                     # 独立试验案例 (如: baby_left_rear_seat)
│   │   ├── data_0001.bin          # 顺序保存的 CIR 原始脉冲数据包
│   │   ├── data_0002.bin
│   │   └── ...
│   └── case2/
│       └── ...
├── batch_2/                       # 物理大批次 (如: empty_cabin_high_temp)
│   └── ...
└── ...
```

### 5.4 Command Line Interface (核心处理类调用规范)

批处理主逻辑实质是通过多进程调用拉起子进程终端，子进程对每个 case 路径执行底层的 `uwb_algo_cpd_UAB.py` 并传入控制参数：

```
python uwb_algo_cpd_UAB.py \
    -p ./config/dubhe_config.yaml \
    -d <data_path> \
    --postfix <label> \
    -s <save_path>
```

#### 5.4.1 Argument Details

- `-p`: 指派具体回归使用的座舱配置文件。
- `-d`: 批处理检索到的待解算当前 Case 数据文件夹路径。
- `-n`: 生成的回归 CSV 前缀附加字，作为区分不同参数版本的标签。
- `-s`: 处理结果的导出位置。

### 5.5 Output Files (输出生成分析报告)

回归运行结束后，将在配置的 `save_path` 对应子文件夹内生成如下分析报表：

**Table 16: Batch Processing Output Files**

| Target Name              | File Format | Detailed Contained Content                                   |
| ------------------------ | ----------- | ------------------------------------------------------------ |
| `output_label.txt`       | Text        | 终端打印流重定向文本。包含中间矩阵对齐计算耗时、加载权重提示等 |
| `label_debug.log`        | Text        | 极高精度的物理运算排查日志（含 CFAR 漏检索引警示、信噪比震荡） |
| `label_info.log`         | Text        | 宏观状态生命周期日志，记录主状态转换                         |
| `decision_res_label.csv` | CSV         | 核心占用判决输出报表。包含各帧对应的占用 5 席状态标志字（Seat_A~E） |
| `tgt_info_label.csv`     | CSV         | 空间点云物理信息报表。包含各帧检测到的目标的物理距离 $R$、极坐标角 $\theta, \phi$ 以及 SNR |
| `Graphics/`              | Folder      | 当配置 `plt_mode > 0` 时，自动导出的各帧 Range-Doppler 能量谱、DBF 方位角度扫描图 |

### 5.6 Running Batch Processing (启动执行)

#### 5.6.1 Basic Usage

在完成物理极极板阵列和路径参数配置后，在安装有开发环境的电脑终端运行：

```
python batch_run_RSP.py
```

终端将输出当前的并发处理进程进度，子核心完成后会在目的路径保存相应的报告文件。

#### 5.6.2 Customizing for Different Projects (项目间移植范例)

若需对不同项目（如不同安装位置、座舱空间几何配置）执行批处理分析，只需配置不同的子 YAML，例如：

```
# 例如需要对 AA 项目后排定制天线执行全量 4 进程多核分析
save_path = r"O:\roadtest\System\Dubhe\AA\CPD\et5\result_Hui\ver4\batchtest"
root_path = r"O:\roadtest\System\Dubhe\AA\CPD\et5\batch1"
config_file = "./config/dubhe_config_AA.yaml"
```

# Glossary (名词/术语表)

*(Refer to Table 1, Table 2, Abbreviations, and Symbol sections for precise system and antenna geometry definitions / 详细的天线、信号处理与车辆映射术语请参阅引言及第一、二章中的符号与缩写清单)*