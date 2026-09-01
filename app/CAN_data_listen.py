import threading
import queue
import struct
from app.device_management import CAN_GetMsg, CAN_SendMsg, DevicePara, init_device
from app.radar_config import *

import time

class CANFrameServer:
    def __init__(self, cfg):
        """
        初始化 CAN 帧服务器
        :param cfg: 传入 config_2x4 模块
        """
        self.config = cfg
        self.is_running = False
        self.listen_thread = None
        
        # 线程安全队列，用于存放标准化后的帧 (bytes)
        self.frame_queue = queue.Queue(maxsize=2000)
        
        # 报文重组缓冲区 (key: sequence_number)
        self.reassembly_buffer = {}
        
        # 单通道字节长度 (32 taps * 4 bytes/tap = 128 bytes)
        self.bytes_per_channel = cfg.FT_LEN * 4
        # 数据流缓冲区：累积 5 个 CAN 帧的数据 (5 * 64 = 320 bytes)
        self.stream_buffer = bytearray()
        self.PACKET_SIZE = cfg.PACKET_SIZE   # 完整的传输单元大小
        self.HEADER_SIZE = cfg.HEADER_SIZE    # UCI(4B) + Payload Header(16B)
        self.CIR_DATA_SIZE = cfg.CIR_DATA_SIZE # 64 taps * 4 bytes (包含 2 个 RX 通道的数据)
        self.UCI_SIGNATURE = cfg.UCI_SIGNATURE
        self.PACKET_CNT = cfg.PACKET_CNT
        # --- 新增：快照组装缓冲区 ---
        # 结构：{ snapshot_id: { sub_idx: packet_data } }
        self.snapshot_assembly = {}
        self.MAX_BUFFERED_SNAPSHOTS = 5 # 限制缓冲区大小，防止内存溢出

    def start(self):
        """启动监听线程"""
        self.is_running = True
        self.listen_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.listen_thread.start()
        print("📡 CAN Frame Server 线程已启动")
    
    def stop(self):
        """停止监听"""
        CAN_SendMsg(0x100, RANGE_STOP_CMD_LEN, range_stop_cmd, 1, 10)
        self.is_running = False
        if self.listen_thread:
            self.listen_thread.join()
        print("📡 CAN Frame Server 已停止")

    def get_frame(self):
        """获取一个标准化帧"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _worker_loop(self):
        """核心循环：读取 CAN -> 重组 UCI -> 切割分发"""
        while self.is_running:
            try:
                can_num = CAN_GetMsg()
                if can_num > 0:
                    for i in range(can_num):
                        msg = DevicePara.CanMsgBuffer[i]
                        # 假设数据报文 ID 为 0x100
                        if msg.ID == 0x100:
                            # 提取 64 字节数据
                            self.stream_buffer.extend(bytes(msg.Data[:64]))
                            # 每凑齐 320 字节（5 帧）处理一次
                            while len(self.stream_buffer) >= self.PACKET_SIZE:
                                # 1. 查找同步头
                                header_pos = self.stream_buffer.find(self.UCI_SIGNATURE)
                                if header_pos == -1:
                                    # 如果没找到头，但缓冲区已经很大了，保留最后三个字节（可能头被截断）
                                    if len(self.stream_buffer) > self.PACKET_SIZE * 2:
                                        del self.stream_buffer[:-3]
                                    break
                                if header_pos > 0:
                                    # 丢弃同步头之前的垃圾数据
                                    del self.stream_buffer[:header_pos]
                                    continue
                                # 2. 此时 buffer[0:4] 确定是同步头，且总长度足够        
                                packet = self.stream_buffer[:self.PACKET_SIZE]
                                del self.stream_buffer[:self.PACKET_SIZE]
                                # 进入组装逻辑，不再直接发送
                                self._assemble_and_verify(packet)
                # 避免 CPU 空转
                time.sleep(0.001)
            except Exception as e:
                print(f"CAN 接收错误: {e}")
    
    def _assemble_and_verify(self, packet):
        """验证快照完整性：只有凑齐一个周期(4个分块)才推入队列"""
        # 1. 解析 Index
        idx = struct.unpack('<H', packet[14:16])[0] #
        snapshot_id = idx // self.PACKET_CNT  # 确定快照 ID (每 self.PACKET_CNT 个 idx 为一组)
        sub_idx = idx % self.PACKET_CNT  # 确定在组内的位置 (0, 1, 2, 3,)
        
        # 2. 存入临时缓冲区
        if snapshot_id not in self.snapshot_assembly:
            self.snapshot_assembly[snapshot_id] = {}

        self.snapshot_assembly[snapshot_id][sub_idx] = packet

        # 3. 检查是否凑齐了该快照的所有 4 个分块
        if len(self.snapshot_assembly[snapshot_id]) == self.PACKET_CNT:
            # 顺序处理这 4 个包并发送给消费者
            for s_idx in range(self.PACKET_CNT):
                full_packet = self.snapshot_assembly[snapshot_id][s_idx]
                self._dispatch_to_consumer(full_packet)
            
            # 处理完毕，清理该快照
            del self.snapshot_assembly[snapshot_id]
        # 4. 内存管理：如果积压的快照过多，丢弃最早的（说明由于丢帧无法凑齐）
        if len(self.snapshot_assembly) > self.MAX_BUFFERED_SNAPSHOTS:
            oldest_id = min(self.snapshot_assembly.keys())
            print(f"⚠️ 快照 {oldest_id} 由于丢帧无法凑齐，已丢弃相关数据。")
            del self.snapshot_assembly[oldest_id]

    def _dispatch_to_consumer(self, packet):
        """原有的标准化封装逻辑，将 320B 块拆分为标准帧推入队列"""
        idx = struct.unpack('<H', packet[14:16])[0]
        cir_raw = packet[20:20+self.CIR_DATA_SIZE] #
        
        cycle_pos = idx % self.config.PACKET_CNT
        tx_list = self.config.TX_LIST
        rx_list = self.config.RX_LIST
        if self.config.MCU_NAME == "Calterah":
            # 映射天线 (TX1: 0,1; TX2: 2,3)
            tx_id = tx_list[0] if cycle_pos < 2 else tx_list[1]
            rxs = rx_list[0:2] if cycle_pos % 2 == 0 else rx_list[2:4]

            for i, rx_id in enumerate(rxs):
                chunk = cir_raw[i*128 : (i+1)*128]
                standard_frame = (
                    self.config.START_SIGN + 
                    struct.pack('BB', tx_id, rx_id) + 
                    chunk + 
                    self.config.STOP_SIGN
                )
                if not self.frame_queue.full():
                    self.frame_queue.put(standard_frame)
        elif self.config.MCU_NAME == "29D6":
            # 1. 获取当前 RX 列表的长度，确保鲁棒性
            num_rx = len(rx_list)
            
            # 2. 计算当前 cycle_pos 对应的 TX 和 RX 索引
            # 使用整除获取 TX 索引，使用取余获取 RX 索引
            tx_idx = cycle_pos // num_rx
            rx_idx = cycle_pos % num_rx
            # 3. 提取具体的 ID
            # 使用 try-except 或 min 防止越界（虽然逻辑上 cycle_pos 不应越界）
            tx_id = tx_list[tx_idx]
            rx_id = rx_list[rx_idx]
            chunk = cir_raw
            standard_frame = (
                self.config.START_SIGN + 
                struct.pack('BB', tx_id, rx_id) + 
                chunk + 
                self.config.STOP_SIGN
            )
            if not self.frame_queue.full():
                self.frame_queue.put(standard_frame)

# ==============================================================================
# 雷达启动流程封装 (来自 main.py)
# ==============================================================================
def radar_hardware_init():
    """执行完整的雷达启动命令序列"""
    print("🚀 正在发送雷达启动配置指令...")
    commands = [
        (0x100, device_reset_cmd, DEVICE_RESET_CMD_LEN),
        (0x100, session_init_cmd, SESSION_INIT_CMD_LEN),
        (0x100, session_set_app_config_cmd, SESSION_SET_APP_CONFIG_CMD_LEN),
        (0x100, session_set_radar_config_cmd, SESSION_SET_RADAR_CONFIG_CMD_LEN),
        (0x100, range_start_cmd, RANGE_START_CMD_LEN),
    ]
    
    for cmd_id, data, dlc in commands:
        CAN_SendMsg(cmd_id, dlc, data, 1, 10)
        time.sleep(0.1)
    
    print("✅ 雷达启动指令发送完毕")

# ==============================================================================
# 监听测试入口
# ==============================================================================
if __name__ == "__main__":
    import config_2x4 as config
    # 1. 硬件初始化与雷达启动
    device_choice = input("请输入CAN设备类型 (ZLGCAN/TOOMOSS, 默认 ZLGCAN): ") or "ZLGCAN"
    
    if radar_hardware_init(device_choice):
        # 2. 启动监听服务
        server = CANFrameServer(config)
        server.start()
        
        print("🔍 正在监听标准化帧，按 Ctrl+C 停止...")
        try:
            while True:
                # 3. 模拟消费者：从队列中获取标准化帧并简单打印
                frame = server.get_frame()
                if frame:
                    # 解析头部信息查看天线对
                    tx_id, rx_id = struct.unpack('BB', frame[4:6])
                    print(f"[{time.strftime('%H:%M:%S')}] 捕获标准帧: "
                          f"TX={tx_id}, RX={rx_id}, 长度={len(frame)} 字节")
                
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\n🛑 正在停止...")
            # 停止雷达发送
            
            server.stop()