# -*- coding: utf-8 -*-
"""
serial_rx_test.py — 串口接收链路自检（下位机 ↔ 上位机 UART）

功能：
    1. 确认下位机是否在持续发送数据、上位机能否正确分帧；
    2. 高精度观测帧间隔（两次帧到达差多少毫秒），判断下位机节奏是否
       稳定、是否偶发丢帧/卡顿；
    3. 输出通道(TX/RX)分布、帧头帧尾校验、CIR 概要等诊断信息。

帧格式（标准帧，与 gui_main.RadarProtocol 一致）：
    START_SIGN(4B FF 00 FF 00) | TX(1B) | RX(1B) | CIR(ft_len*4B, I/Q int16 交错) | STOP_SIGN(4B F0 00 F0 00)
    默认 ft_len=32 → 每帧 138 字节；1T4R 下 4 帧 = 1 个快照。

用法：
    python test/serial_rx_test.py                          # 参数自动读取 config_save.json
    python test/serial_rx_test.py --port COM12 --baud 1000000
    python test/serial_rx_test.py --list                   # 仅列出可用串口
    python test/serial_rx_test.py --duration 10            # 测 10 秒后自动退出
    python test/serial_rx_test.py --timing                 # 逐帧打印到达时刻与帧间隔
    python test/serial_rx_test.py --verbose                # 每秒附加 CIR 概要

退出码：
    0 = 收到完整帧，链路正常     2 = 完全无数据     3 = 有数据但无法分帧
"""
import argparse
import json
import struct
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frame_sync import FrameSyncBuffer  # noqa: E402  复用根目录分帧器

START_SIGN = b"\xff\x00\xff\x00"
STOP_SIGN = b"\xf0\x00\xf0\x00"
BITS_PER_BYTE = 10  # UART: 1 start + 8 data + 1 stop
DEFAULT_TX, DEFAULT_RX_FIRST = 2, 4  # 快照起始通道（与 config_save.json 的 1T4R 一致）

try:  # Windows 控制台可能非 UTF-8，避免中文打印崩溃
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ==============================================================================
# 工具函数
# ==============================================================================
def load_default_port_config() -> Tuple[str, int, int]:
    """从 config_save.json 读取串口配置（与 GUI 保持一致）。"""
    cfg_path = ROOT / "config_save.json"
    if cfg_path.exists():
        try:
            rc = json.loads(cfg_path.read_text(encoding="utf-8")).get("radar_config", {})
            return (
                rc.get("serial_port", ""),
                int(rc.get("baud_rate", 1_000_000)),
                int(rc.get("ft_len", 32)),
            )
        except Exception:
            pass
    return "", 1_000_000, 32


def parse_frame(frame: bytes, ft_len: int) -> Tuple[int, int, Tuple[int, ...]]:
    """解析标准帧 → (tx, rx, s16)。s16: 2*ft_len 个 int16（I/Q 交错）。"""
    s16 = struct.unpack("<%dh" % (ft_len * 2), frame[6:-4])
    return frame[4], frame[5], s16


def cir_summary(s16: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """CIR 概要 → (max|I|, max|Q|, 非零采样数, 总采样数)。"""
    max_i = max(abs(s16[i]) for i in range(0, len(s16), 2))
    max_q = max(abs(s16[i]) for i in range(1, len(s16), 2))
    return max_i, max_q, sum(1 for v in s16 if v != 0), len(s16)


def dump_hex(data: bytes, n: int = 64) -> None:
    """打印原始字节（hex + ASCII），供协议不匹配时排查。"""
    chunk = data[:n]
    print("    最近原始字节 hex: %s" % chunk.hex(" "))
    print("    对应 ascii: %s" % "".join(chr(b) if 32 <= b < 127 else "." for b in chunk))


# ==============================================================================
# 时序统计
# ==============================================================================
@dataclass
class IntervalStats:
    """收集间隔序列：实时窗口估计 + 整体分位数统计。"""

    window: int = 64
    _all: List[float] = field(default_factory=list)
    _recent: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.window)

    def add(self, dt: float) -> None:
        self._all.append(dt)
        self._recent.append(dt)

    @property
    def count(self) -> int:
        return len(self._all)

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(len(ordered) * pct))]

    def recent_median_ms(self) -> Optional[float]:
        """最近窗口间隔中位数（ms），实时速率估计用，免疫启动空窗。"""
        return self._percentile(list(self._recent), 0.5) * 1000.0 if self._recent else None

    def recent_std_ms(self) -> Optional[float]:
        """最近窗口间隔标准差（ms），衡量节奏抖动。"""
        if len(self._recent) < 2:
            return None
        vals = list(self._recent)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return var ** 0.5 * 1000.0

    def summarize(self, label: str) -> str:
        """整体统计摘要行（n/min/avg/max/std/p50/p95/p99）。"""
        vals = self._all
        if not vals:
            return f"  {label:<10} (无样本)"
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        ms = lambda v: v * 1000.0  # noqa: E731
        return (
            f"  {label:<10} n={len(vals):<6} min={ms(min(vals)):8.3f}ms "
            f"avg={ms(mean):8.3f}ms max={ms(max(vals)):8.3f}ms "
            f"std={ms(var ** 0.5):7.3f}ms  p50={ms(self._percentile(vals, 0.5)):8.3f}ms "
            f"p95={ms(self._percentile(vals, 0.95)):8.3f}ms "
            f"p99={ms(self._percentile(vals, 0.99)):8.3f}ms"
        )


# ==============================================================================
# 会话状态
# ==============================================================================
@dataclass
class Session:
    """一次接收会话的累计状态与计数器。"""

    fs: FrameSyncBuffer
    first_pair: Tuple[int, int]
    chan_counts: Counter = field(default_factory=Counter)
    recent_cir: Dict[Tuple[int, int], Tuple[int, int, int, int]] = field(default_factory=dict)
    frame_gaps: IntervalStats = field(default_factory=IntervalStats)
    snap_gaps: IntervalStats = field(default_factory=IntervalStats)
    bytes_total: int = 0
    frames_total: int = 0
    bad_tail: int = 0
    raw_hist: bytes = b""
    idle_warned: bool = False
    no_frame_warned: bool = False

    # 上一帧/上一快照的到达时刻（perf_counter 秒）
    _last_frame_t: Optional[float] = field(default=None, init=False, repr=False)
    _last_snap_t: Optional[float] = field(default=None, init=False, repr=False)

    def on_frame(self, fr: bytes, arrive_t: float, ft_len: int) -> Optional[float]:
        """登记一个完整帧，返回它与上一帧的间隔（秒）；首帧返回 None。

        同时维护：总数/帧尾校验/通道计数/CIR 概要，以及帧间隔、
        快照周期（首通道帧出现即视为新快照）统计。
        """
        self.frames_total += 1
        if not fr.endswith(STOP_SIGN):
            self.bad_tail += 1

        gap = None if self._last_frame_t is None else arrive_t - self._last_frame_t
        if gap is not None:
            self.frame_gaps.add(gap)
        self._last_frame_t = arrive_t

        tx, rx, s16 = parse_frame(fr, ft_len)
        self.chan_counts[(tx, rx)] += 1
        self.recent_cir[(tx, rx)] = cir_summary(s16)

        if (tx, rx) == self.first_pair:  # 新快照开始
            if self._last_snap_t is not None:
                self.snap_gaps.add(arrive_t - self._last_snap_t)
            self._last_snap_t = arrive_t

        return gap


# ==============================================================================
# 接收主循环
# ==============================================================================
def run_session(ser, args, baud: int, ft_len: int) -> Session:
    frame_len = 4 + 2 + ft_len * 4 + 4
    frame_air_time = frame_len * BITS_PER_BYTE / baud  # 单帧在线传输耗时（秒）
    first_pair = (args.tx or DEFAULT_TX, args.rx_first or DEFAULT_RX_FIRST)

    sess = Session(fs=FrameSyncBuffer(START_SIGN, frame_len), first_pair=first_pair)
    t_start = time.perf_counter()
    last_report = t_start

    print(f"[OK] 开始接收 @ {baud} baud，帧长 {frame_len}B，Ctrl+C 结束\n")

    try:
        while True:
            # ---- 1. 读取 ----
            n = ser.in_waiting
            chunk = ser.read(n) if n else b""
            if not chunk:
                time.sleep(0.005)
                continue
            recv_t = time.perf_counter()  # 本批数据到达时刻
            sess.bytes_total += len(chunk)
            sess.raw_hist = (sess.raw_hist + chunk)[-64:]

            # ---- 2. 分帧；同批 k 帧按线速近似分摊到达时刻 ----
            frames = sess.fs.feed(chunk)
            k = len(frames)
            for i, fr in enumerate(frames):
                arrive_t = recv_t - (k - 1 - i) * frame_air_time
                gap = sess.on_frame(fr, arrive_t, ft_len)
                if args.timing:
                    gap_txt = f"{gap * 1000.0:8.3f} ms" if gap is not None else "   (首帧)"
                    tx, rx, _ = parse_frame(fr, ft_len)
                    print(f"  [t+{arrive_t - t_start:9.6f}s] TX{tx}-RX{rx}  间隔 {gap_txt}",
                          flush=True)

            # ---- 3. 每秒状态行 ----
            now = time.perf_counter()
            if now - last_report >= 1.0:
                last_report = now
                _print_status(sess, args)

            # ---- 4. 诊断提示（各一次） ----
            _warn_if_needed(sess, now - t_start)

            # ---- 5. 结束 ----
            if args.duration > 0 and now - t_start >= args.duration:
                print(f"\n达到设定时长 {args.duration:.0f}s，停止。")
                break

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止接收。")

    return sess


def _print_status(sess: Session, args) -> None:
    """每秒一行：帧数 / 速率 / 间隔抖动 / 通道分布。"""
    snap_med = sess.snap_gaps.recent_median_ms()
    rate_txt = f"{1000.0 / snap_med:6.1f} 快照/s" if snap_med else "快照周期 --"
    frame_med = sess.frame_gaps.recent_median_ms()
    frame_std = sess.frame_gaps.recent_std_ms()
    gap_txt = (f"帧间隔中位 {frame_med:.3f}ms σ{frame_std:.3f}ms"
               if frame_med is not None else "帧间隔 --")
    chan_txt = " ".join(f"TX{t}-RX{r}:{c}" for (t, r), c in sorted(sess.chan_counts.items()))
    print(f"[{time.strftime('%H:%M:%S')}] 帧 {sess.frames_total:>6} | {rate_txt} | "
          f"{gap_txt} | {chan_txt or '(暂无)'}", flush=True)
    if args.verbose and sess.recent_cir:
        for (t, r), (mi, mq, nz, tot) in list(sess.recent_cir.items())[-4:]:
            print(f"    TX{t}-RX{r}: max|I|={mi} max|Q|={mq} 非零 {nz}/{tot}")


def _warn_if_needed(sess: Session, elapsed: float) -> None:
    """无数据 / 无法分帧 两类问题的提示（各打印一次）。"""
    if sess.bytes_total == 0 and not sess.idle_warned and elapsed >= 3.0:
        print("\n[警告] 已 3 秒未收到任何字节！请检查：")
        print("   - 下位机是否上电并开始发送？TX/RX 是否交叉接线并共地？")
        print("   - 波特率是否一致？串口是否被 GUI/其他程序占用？驱动是否正常？\n", flush=True)
        sess.idle_warned = True
    if (sess.bytes_total > 0 and sess.frames_total == 0
            and not sess.no_frame_warned and elapsed >= 3.0):
        print("\n[警告] 已收到字节但解不出完整帧！帧格式可能与预期不符，")
        print(f"  分帧器期望: 帧长 {sess.fs._frame_len}B、帧头 {START_SIGN.hex()}、"
              f"帧尾 {STOP_SIGN.hex()}。")
        dump_hex(sess.raw_hist)
        print("  可能: 协议不是本工程标准帧 / ft_len 不对 / 接线松动丢字节\n", flush=True)
        sess.no_frame_warned = True


# ==============================================================================
# 汇总报告
# ==============================================================================
def _report(sess: Session) -> int:
    """打印汇总（含时序统计）并返回退出码。"""
    print("\n" + "=" * 76)
    print("汇总")
    print(f"  总字节 {sess.bytes_total} | 完整帧 {sess.frames_total} | "
          f"帧尾异常 {sess.bad_tail} | 整除 {sess.bytes_total / sess.fs._frame_len:.3f}"
          if sess.frames_total else
          f"  总字节 {sess.bytes_total} | 完整帧 {sess.frames_total} | 帧尾异常 {sess.bad_tail}")
    if sess.chan_counts:
        print("  通道分布: " + ", ".join(
            f"TX{t}-RX{r}:{c}" for (t, r), c in sorted(sess.chan_counts.items())))
        if sess.recent_cir:
            (t, r), (mi, mq, nz, tot) = list(sess.recent_cir.items())[0]
            print(f"  最近一帧样例 TX{t}-RX{r}: max|I|={mi} max|Q|={mq} 非零 {nz}/{tot}")

    if sess.frame_gaps.count:
        print("  时序统计（4 帧 = 1 快照；帧间隔期望≈25ms，快照周期≈100ms）:")
        print(sess.frame_gaps.summarize("帧间隔"))
        if sess.snap_gaps.count:
            print(sess.snap_gaps.summarize("快照周期"))
        gaps = sess.frame_gaps._all
        p50 = sorted(gaps)[len(gaps) // 2]
        anomalies = [g for g in gaps if g > 1.5 * p50]
        if anomalies:
            print(f"  异常帧间隔(>1.5×p50={p50 * 1000:.3f}ms): {len(anomalies)} 次，"
                  f"最大 {max(anomalies) * 1000:.3f}ms —— 疑似偶发丢帧/卡顿")
        else:
            print("  异常帧间隔: 0 次 —— 节奏稳定，无丢帧迹象")

    if sess.frames_total > 0:
        print("\n[结论] 下位机发送正常，上位机接收/分帧正常 ✔")
        if sess.bad_tail:
            print(f"  (注意: {sess.bad_tail} 帧帧尾异常，整体链路仍可用)")
        return 0
    if sess.bytes_total == 0:
        print("\n[结论] 未收到任何数据 ✘  检查供电/接线/波特率/端口占用（见上方警告）")
        return 2
    print("\n[结论] 收到数据但无法按标准帧解析 ✘  原始字节见上方 hex 输出")
    return 3


# ==============================================================================
# 入口
# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="串口接收链路自检（下位机 ↔ 上位机）")
    ap.add_argument("--port", help="串口号，如 COM12（默认读 config_save.json）")
    ap.add_argument("--baud", type=int, help="波特率（默认 1000000）")
    ap.add_argument("--ft-len", type=int, help="每帧 CIR 复数点数（默认 32）")
    ap.add_argument("--tx", type=int, help="快照起始通道 TX 号（默认 2）")
    ap.add_argument("--rx-first", type=int, help="快照起始通道 RX 号（默认 4）")
    ap.add_argument("--duration", type=float, default=0, help="测试时长秒数，0=直到 Ctrl+C")
    ap.add_argument("--list", action="store_true", help="仅列出可用串口")
    ap.add_argument("--timing", action="store_true", help="逐帧打印到达时刻与帧间隔")
    ap.add_argument("--verbose", action="store_true", help="每秒附加 CIR 概要")
    args = ap.parse_args()

    # ---- 串口枚举 ----
    try:
        from serial.tools import list_ports
        ports = list(list_ports.comports())
    except ImportError:
        print("[错误] 未安装 pyserial，请先: pip install pyserial")
        return 2
    except Exception:
        ports = []
    if args.list:
        if not ports:
            print("未发现任何串口设备（确认 USB 转串口已插入且驱动正常）")
            return 2
        for p in ports:
            print(f"  {p.device:<10} {p.description}")
        return 0

    # ---- 参数补齐 ----
    cfg_port, cfg_baud, cfg_ft = load_default_port_config()
    port = args.port or cfg_port
    baud = args.baud or cfg_baud
    ft_len = args.ft_len or cfg_ft
    if not port:
        print("[错误] 未指定串口且 config_save.json 无 serial_port，请用 --port COMxx。")
        if ports:
            print("可用串口:")
            for p in ports:
                print(f"  {p.device:<10} {p.description}")
        return 2

    frame_len = 4 + 2 + ft_len * 4 + 4
    print("=" * 76)
    print("串口接收链路自检")
    print(f"  串口 {port} @ {baud} baud | 帧长 {frame_len}B (ft_len={ft_len})")
    print(f"  快照起始通道: TX{args.tx or DEFAULT_TX}-RX{args.rx_first or DEFAULT_RX_FIRST}"
          + (f" | 逐帧 timing" if args.timing else ""))
    print(f"  时长: {'无限(Ctrl+C)' if not args.duration else f'{args.duration:.0f}s'}")
    print("=" * 76)

    # ---- 打开串口 ----
    try:
        import serial
        ser = serial.Serial(port=port, baudrate=baud, timeout=0.2)
    except Exception as e:
        print(f"[错误] 打开串口 {port} 失败: {e}")
        print("  可能: 端口不存在 / 被 GUI 占用 / 驱动异常")
        return 2

    try:
        sess = run_session(ser, args, baud, ft_len)
    finally:
        ser.close()
    return _report(sess)


if __name__ == "__main__":
    sys.exit(main())
