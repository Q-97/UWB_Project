# UDP_data_listen.py

import socket
import threading
import queue
import time
import struct

class UDPFrameServer:
    def __init__(self, config):
        """
        初始化 UDP 帧服务器
        :param config: 传入 config_2x4 模块，用于获取协议定义
        """
        self.ip = config.UDP_IP
        self.port = config.UDP_PORT
        self.config = config
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        # 设置超时以便线程能响应停止信号
        self.sock.settimeout(1.0) 
        
        self.is_running = False
        self.listen_thread = None
        
        # 线程安全队列，用于存放打包好的标准帧 (bytes)
        # RadarHost 将从这里取数据，就像从串口缓冲区取数据一样
        self.frame_queue = queue.Queue(maxsize=1000)
        
        # 计算单通道数据长度
        self.bytes_per_channel = config.FT_LEN * 4 

    def start(self):
        """启动监听线程"""
        self.is_running = True
        self.listen_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.listen_thread.start()
        print(f"📡 UDP Frame Server 启动于 {self.ip}:{self.port}")

    def stop(self):
        """停止监听"""
        self.is_running = False
        if self.listen_thread:
            self.listen_thread.join()
        self.sock.close()
        print("📡 UDP Frame Server 已停止")

    def get_frame(self):
        """
        供外部调用的接口：获取一个打包好的帧
        :return: frame bytes or None (if empty)
        """
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _worker_loop(self):
        """后台工作循环：接收 -> 切分 -> 打包 -> 入队"""
        # 预计算总数据大小，用于校验
        total_tx = len(self.config.UDP_TX_LIST)
        total_rx = len(self.config.UDP_RX_LIST)
        expected_size = total_tx * total_rx * self.bytes_per_channel

        while self.is_running:
            try:
                data, addr = self.sock.recvfrom(4096)
                
                # 1. 简单校验
                if len(data) != expected_size:
                    # print(f"⚠️ 数据包大小不匹配: {len(data)} != {expected_size}")
                    continue

                # 2. 切分并打包
                offset = 0
                
                # 假设数据顺序：TX1(RX4,5,6,7) -> TX2(RX4,5,6,7)
                for tx_id in self.config.UDP_TX_LIST:
                    for rx_id in self.config.UDP_RX_LIST:
                        
                        # 提取当前通道的 CIR 数据
                        cir_chunk = data[offset : offset + self.bytes_per_channel]
                        # if b'\xff\x7f' in cir_chunk:
                        #     print(f"UDP Server 检测到 TX{tx_id}-RX{rx_id} 原始数据饱和 (32767)!")


                        offset += self.bytes_per_channel
                        
                        # 构建标准帧
                        # 格式: START + TX_BYTE + RX_BYTE + CIR_DATA + STOP
                        # struct.pack('BB') 将整数转为无符号字节
                        antenna_info = struct.pack('BB', tx_id, rx_id)
                        
                        full_frame = (
                            self.config.START_SIGN + 
                            antenna_info + 
                            cir_chunk + 
                            self.config.STOP_SIGN
                        )
                        
                        # 放入队列
                        if not self.frame_queue.full():
                            self.frame_queue.put(full_frame)
                        else:
                            # 队列满了，丢弃旧的以保持实时性
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put(full_frame)
                            except:
                                pass

            except socket.timeout:
                continue
            except Exception as e:
                print(f"UDP Server Error: {e}")
                time.sleep(0.1)

# 测试代码
if __name__ == "__main__":
    import config_2x4 as config
    server = UDPFrameServer(config)
    server.start()
    try:
        while True:
            frame = server.get_frame()
            if frame:
                print(f"Got frame: len={len(frame)}, hex={frame[:10].hex()}...")
            time.sleep(0.01)
    except KeyboardInterrupt:
        server.stop()