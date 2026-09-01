"""协议层：标准帧解析（RadarProtocol）与配置桥接（ConfigAdapter）。

原 gui_main.py 拆分产物（阶段 1：纯搬迁，行为不变）。
"""
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

