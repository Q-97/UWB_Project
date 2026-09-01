"""协议层：标准帧解析（RadarProtocol）与配置桥接（ConfigAdapter）。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
阶段 3 新增 FrameParser 注册表：为新的下位机帧格式（V2）提供扩展插槽，
V1 解析路径（RadarProtocol.parse_frame）保持不变。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# ==============================================================================
# 2. Protocol & Config
# ==============================================================================
class RadarProtocol:
    START_SIGN = b'\xff\x00\xff\x00'; STOP_SIGN = b'\xf0\x00\xf0\x00'
    HEADER_LEN = 4; FOOTER_LEN = 4; ANTENNA_INFO_LEN = 2
    FT_LEN = 32; CIR_DATA_LEN = 128; FRAME_LEN = 138

    @classmethod
    def update_protocol(cls, ft_len, start_sign=None, stop_sign=None):
        cls.FT_LEN = ft_len
        cls.CIR_DATA_LEN = ft_len * 4
        cls.FRAME_LEN = cls.HEADER_LEN + cls.ANTENNA_INFO_LEN + cls.CIR_DATA_LEN + cls.FOOTER_LEN
        if start_sign is not None:
            cls.START_SIGN = start_sign
        if stop_sign is not None:
            cls.STOP_SIGN = stop_sign

    @staticmethod
    def parse_frame(frame_bytes: bytes):
        if len(frame_bytes) != RadarProtocol.FRAME_LEN: return None
        if not frame_bytes.startswith(RadarProtocol.START_SIGN): return None
        tx = frame_bytes[RadarProtocol.HEADER_LEN]; rx = frame_bytes[RadarProtocol.HEADER_LEN+1]
        cir = frame_bytes[RadarProtocol.HEADER_LEN+2 : -RadarProtocol.FOOTER_LEN]
        s16 = np.frombuffer(cir, dtype=np.int16)
        c_data = s16[0::2].astype(np.float32) + 1j * s16[1::2].astype(np.float32)
        return tx, rx, c_data

class ConfigAdapter:
    def __init__(self, dynamic_config):
        # UDP 字段
        self.UDP_IP = dynamic_config.udp_ip; self.UDP_PORT = dynamic_config.udp_port
        self.UDP_TX_LIST = dynamic_config.udp_tx_list; self.UDP_RX_LIST = dynamic_config.udp_rx_list
        self.FT_LEN = dynamic_config.ft_len
        self.START_SIGN = RadarProtocol.START_SIGN; self.STOP_SIGN = RadarProtocol.STOP_SIGN
        # CAN 字段 (CANFrameServer 使用不同命名，桥接过来)
        self.TX_LIST = dynamic_config.udp_tx_list
        self.RX_LIST = dynamic_config.udp_rx_list
        self.MCU_NAME = dynamic_config.mcu_name
        self.PACKET_SIZE = dynamic_config.can_packet_size
        self.HEADER_SIZE = dynamic_config.can_header_size
        self.CIR_DATA_SIZE = dynamic_config.can_cir_data_size
        self.UCI_SIGNATURE = bytes.fromhex(dynamic_config.can_uci_signature)
        self.PACKET_CNT = dynamic_config.can_packet_cnt


# ==============================================================================
# FrameParser 注册表（阶段 3：协议插件化，为 V2 帧格式预留插槽）
# ==============================================================================
@dataclass
class ParsedFrame:
    """统一解析结果。"""
    tx: int
    rx: int
    cir: np.ndarray        # complex64, len = FT_LEN
    raw: bytes = b""


class FrameParser(ABC):
    """协议解析器抽象：新协议（如 V2 帧格式）实现后 register_parser 即可。"""

    name: str = "base"

    @property
    @abstractmethod
    def frame_len(self) -> int:
        """该协议的单帧字节长（供数据源定长切帧）。"""

    @abstractmethod
    def can_parse(self, frame: bytes) -> bool:
        """是否能解析该帧（建议只查魔数/版本字段）。"""

    @abstractmethod
    def parse(self, frame: bytes) -> ParsedFrame:
        """解析一帧，非法时抛异常。"""


class UwbV1Parser(FrameParser):
    """现有 V1 标准帧：START(4B) + TX(1B) + RX(1B) + CIR + STOP(4B)。"""

    name = "uwb_v1"

    @property
    def frame_len(self) -> int:
        return RadarProtocol.FRAME_LEN

    def can_parse(self, frame: bytes) -> bool:
        return len(frame) == self.frame_len and frame.startswith(RadarProtocol.START_SIGN)

    def parse(self, frame: bytes) -> ParsedFrame:
        result = RadarProtocol.parse_frame(frame)
        if result is None:
            raise ValueError("invalid V1 frame")
        tx, rx, cir = result
        return ParsedFrame(tx=tx, rx=rx, cir=cir, raw=bytes(frame))


_PARSERS: list = [UwbV1Parser()]


def register_parser(parser: FrameParser) -> None:
    """注册新协议解析器（V2 帧格式定稿后一行注册即可）。"""
    _PARSERS.append(parser)


def get_parsers():
    """当前已注册的解析器列表（副本）。"""
    return list(_PARSERS)


def parse_frame(frame: bytes):
    """按注册表顺序尝试解析；全部失败返回 None（保持旧接口语义）。"""
    for parser in _PARSERS:
        if parser.can_parse(frame):
            try:
                return parser.parse(frame)
            except Exception:
                return None
    return None

