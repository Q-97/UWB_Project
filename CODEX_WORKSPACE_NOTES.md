# Codex 工作区说明

本文记录本仓库在 Codex 环境中的已知问题和推荐工作流。开启新对话后，请先让 Codex 阅读此文件，再开始修改或验证代码。

## 本次为什么耗时较长

功能本身只涉及 `gui_main.py` 中的 RA-OCCUPANCY 参数、按钮绘制和窗口过滤。额外耗时主要来自以下环境问题：

1. PowerShell 直接使用 `Get-Content gui_main.py` 时，UTF-8 中文注释显示成乱码。把乱码文本用作 `apply_patch` 上下文后，补丁多次匹配失败。
2. 系统 PATH 中没有可用的 `python`。`py.exe` 虽然存在，但执行 `py -3` 会报告 `No installed Python found!`。
3. Codex bundled Python 可以运行标准库检查，但没有 `matplotlib`，所以不能直接 `import gui_main` 或启动完整 GUI。
4. `python -m py_compile gui_main.py` 需要写入 `__pycache__`，本次因已有缓存文件权限问题返回 `WinError 5`。
5. `gui_main.py` 当前工作副本使用 CRLF，但 `.gitattributes` 指定 `eol=lf`。`apply_patch` 新增的局部内容使用 LF，曾造成混合换行和较大的无意义 diff。
6. 仓库原本就是 dirty worktree，`gui_main.py` 同时存在 staged 和 unstaged 修改。验证时必须区分用户已有修改，不能回退或覆盖。

## 环境事实

- 仓库路径：`D:\code\cpd\uwb-occupancy-detection`
- Shell：PowerShell
- 源文件编码：UTF-8，无 BOM
- `gui_main.py` 当前工作副本换行：CRLF
- Git 属性：`* text=auto eol=lf`
- 不要默认使用 `python` 或 `py -3`。
- 需要 Python 时，先调用 Codex 的 workspace dependency loader，使用它返回的 Python 绝对路径。不要硬编码 bundle 版本路径，因为后续版本可能变化。
- Bundled Python 适合 `ast` 等标准库验证，不代表它具备项目运行依赖。

## 推荐的最短工作流

### 1. 修改前先确认工作区

```powershell
git status --short
git diff -- gui_main.py
git diff --cached -- gui_main.py
```

不要回退不属于当前任务的改动。`MM gui_main.py` 表示该文件同时存在 staged 和 unstaged 修改。

### 2. 搜索和读取代码

优先使用 `rg`。需要 PowerShell 分段读取时必须显式指定 UTF-8：

```powershell
rg -n "RA-OCCUPANCY|oa_filter|ra_occ_oa" gui_main.py
$lines = Get-Content -Encoding UTF8 gui_main.py
$lines[3550..3850]
```

不要把乱码输出复制到补丁上下文。修改 `gui_main.py` 时，优先使用函数名、字典 key 等 ASCII 行作为 `apply_patch` 锚点。

### 3. Python 语法验证

先通过 workspace dependency loader 获取 bundled Python 路径，然后用该路径执行只读 AST 检查：

```powershell
& '<workspace dependency loader 返回的 python.exe>' -c "import ast; ast.parse(open('gui_main.py', encoding='utf-8').read()); print('syntax ok')"
```

这条命令不会导入 `matplotlib`，也不会写 `__pycache__`。除非已找到项目实际使用且依赖完整的 Python 环境，否则不要反复尝试以下命令：

```powershell
python -m py_compile gui_main.py
py -3 -m py_compile gui_main.py
python -c "import gui_main"
```

完整 GUI 验证需要一个安装了 `matplotlib`、`numpy`、`scipy`、`pyserial` 及项目其他依赖的 Python 环境。当前 Codex bundled Python 不满足该条件。

### 4. 换行符检查

先检查文件是否出现混合换行：

```powershell
$bytes = [System.IO.File]::ReadAllBytes((Resolve-Path gui_main.py))
$text = [System.Text.Encoding]::UTF8.GetString($bytes)
$crlf = ([regex]::Matches($text, "`r`n")).Count
$lf = ([regex]::Matches($text, "(?<!`r)`n")).Count
"CRLF=$crlf LF=$lf"
```

如果修改前文件是 CRLF，而补丁后出现少量独立 LF，只将独立 LF 恢复为 CRLF；不要无理由重写其他内容：

```powershell
$path = (Resolve-Path gui_main.py).Path
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$content = [System.IO.File]::ReadAllText($path, $utf8NoBom)
$content = [System.Text.RegularExpressions.Regex]::Replace($content, "(?<!`r)`n", "`r`n")
[System.IO.File]::WriteAllText($path, $content, $utf8NoBom)
```

由于工作副本是 CRLF，diff 空白检查使用：

```powershell
git -c core.whitespace=cr-at-eol diff --check
```

### 5. 最终核对

```powershell
git diff --stat -- gui_main.py
git diff -- gui_main.py
git -c core.whitespace=cr-at-eol diff --check
```

确认 diff 只包含当前功能，并再次运行 AST 语法检查。完整 GUI 没有实际启动时，应在交付说明中明确写出，不要把 AST 检查描述成 GUI 运行验证。

## RA-OCCUPANCY 当前窗口过滤功能

当前 `gui_main.py` 已加入以下功能：

- `oa_filter_window_size`：滑动窗口长度，默认 `5`。
- `oa_filter_in_threshold`：窗口内判定 IN 所需次数，默认 `3`。
- 新按钮位于原始 IN/OUT 按钮左侧。
- 新按钮直接读取与原按钮相同的 `oa_label`，不读取原按钮 patch 或 text，因此删除原按钮显示后仍能工作。
- 必须收满一个完整窗口后才允许输出过滤后的 IN。
- 窗口不足阈值时显示 OUT；原始状态为 EMPTY 时保持 EMPTY 显示。
- 布局重建、重新启动和切换回放文件时会清空过滤窗口。
- 两个新参数位于 `DEFAULT_ALGO_PARAMS['RA-OCCUPANCY']`，会自动出现在现有 `Params` 对话框中，无需另建设置窗口。

相关位置不要依赖固定行号，使用以下命令定位：

```powershell
rg -n "oa_filter_window_size|oa_filter_in_threshold|def _update_ra_occ_oa_filter|ra_occ_oa_filter_button" gui_main.py
```

## 新对话建议开场指令

可以在新对话中直接说明：

> 请先完整阅读仓库根目录的 `CODEX_WORKSPACE_NOTES.md`，按其中的编码、Python 验证、换行符和 dirty worktree 注意事项执行，再处理我的任务。

