# -*- coding: utf-8 -*-
"""
frame_sync.py — 字节流帧同步器

把连续的字节流切分成完整帧（找帧头 + 按定长切帧）。
与具体传输方式解耦：串口、CAN 重组后的字节流、文件流均可复用。

用法::

    fs = FrameSyncBuffer(b'\\xff\\x00\\xff\\x00', 138)
    frames = fs.feed(ser.read(ser.in_waiting))   # 返回 0..N 个完整帧
"""
from typing import List


class FrameSyncBuffer:
    """帧同步缓冲：喂入字节流，吐出完整帧列表。"""

    def __init__(self, start_sign: bytes, frame_len: int):
        """
        :param start_sign: 帧头（如 b'\\xff\\x00\\xff\\x00'）
        :param frame_len:  帧总长（定长帧，含帧头与帧尾）
        """
        self._start = start_sign
        self._frame_len = frame_len
        self._buf = b''

    def feed(self, chunk: bytes) -> List[bytes]:
        """喂入一段字节流，返回其中完整的帧（可能为空列表）。

        chunk 可为空：此时仅消化缓冲中已积压的完整帧。
        缓冲中找不到完整帧头时，保留尾部最多 ``len(start_sign)-1`` 字节，
        防止帧头被读取边界截断导致永久失步。
        """
        frames: List[bytes] = []
        self._buf += chunk

        while True:
            start = self._buf.find(self._start)
            if start < 0:
                # 无完整帧头：只保留尾部可能被截断的帧头片段，其余垃圾丢弃
                keep = min(len(self._start) - 1, len(self._buf))
                self._buf = self._buf[-keep:] if keep else b''
                break
            if start > 0:
                # 丢弃帧头前的垃圾字节
                self._buf = self._buf[start:]
            if len(self._buf) < self._frame_len:
                break  # 帧头在，但帧不完整，等下一块数据
            frames.append(self._buf[:self._frame_len])
            self._buf = self._buf[self._frame_len:]
        return frames

    def reset(self) -> None:
        """清空内部缓冲（如协议参数变化后重建对齐）。"""
        self._buf = b''
