# coding:utf-8
from ctypes import *
import platform
from time import sleep
from app.comm.usb_device import *
from app.comm.usb2canfd import *
import random
import struct

FRE_CHILD = 25
FRE_ADULT = 20
PARA = {
    0x00: 'Cir_Offset',
    0x01: 'Tx_Power',
    0x02: 'AMP_Start_Path',
    0x03: 'AMP_Stop_Path',
    0x04: 'AMP_Thresh',
    0x05: 'PHS_Start_Path',
    0x06: 'PHS_Stop_Path',
    0x07: 'PHS_Thresh',
    0x08: 'Motion_Start_Path',
    0x09: 'Motion_Stop_Path',
    0x0A: 'Motion_Thresh'
}
NUM_TYPE = {
    0x00: 1,
    0x01: -1,
    0x02: 0.1,
    0x03: 0.01,
    0x04: 1.0
}


class CPD_data:
    def __init__(self):
        self.Status = ""
        self.Energy = ""
        self.Phase = ""
        self.Amp = ""
        self.Frequency = ""
        self.fps = ""
        self.cir_a8 = ""

    def set_status(self, byte1):
        if byte1 == 0:
            self.Status = "None"

        elif byte1 == 1:
            self.Status = "Breathing"
        else:
            self.Status = 'Moving'

    def set_target(self, byte6):
        if byte6 == 0:
            self.Target = "Adult"
        elif byte6 == 1:
            self.Target = "Child"
        elif byte6 == 2:
            self.Target = "Not Sure"
        elif byte6 == 3:
            self.Target = "None"

    def set_detection(self, byte7):
        if byte7 == 0:
            self.Detection = "Initial value"
        elif byte7 == 1:
            self.Detection = "Not detected"
        elif byte7 == 2:
            self.Detection = "Detecting"


def canfd_init():
    CAN1 = 0
    CAN2 = 1
    DevHandles = (c_uint * 2)()
    # 扫描设备并将设备号存放到设备号数组中
    ret = USB_ScanDevice(byref(DevHandles))
    if (ret == 0):
        print("No device connected!")
        exit()
    else:
        print("Have %d device connected!" % ret)
    # 打开设备
    ret = USB_OpenDevice(DevHandles[0])
    if (bool(ret)):
        print("Open device success!")
    else:
        print("Open device faild!")
        exit()
    # 获取设备固件信息
    USB2XXXInfo = DEVICE_INFO()
    USB2XXXFunctionString = (c_char * 256)()
    ret = DEV_GetDeviceInfo(DevHandles[0], byref(USB2XXXInfo), byref(USB2XXXFunctionString))
    if (bool(ret)):
        print("USB2XXX device infomation:")
        print("--Firmware Name: %s" % bytes(USB2XXXInfo.FirmwareName).decode('ascii'))
        print("--Firmware Version: v%d.%d.%d" % (
        (USB2XXXInfo.FirmwareVersion >> 24) & 0xFF, (USB2XXXInfo.FirmwareVersion >> 16) & 0xFF,
        USB2XXXInfo.FirmwareVersion & 0xFFFF))
        print("--Hardware Version: v%d.%d.%d" % (
        (USB2XXXInfo.HardwareVersion >> 24) & 0xFF, (USB2XXXInfo.HardwareVersion >> 16) & 0xFF,
        USB2XXXInfo.HardwareVersion & 0xFFFF))
        print("--Build Date: %s" % bytes(USB2XXXInfo.BuildDate).decode('ascii'))
        print("--Serial Number: ", end='')
        for i in range(0, len(USB2XXXInfo.SerialNumber)):
            print("%08X" % USB2XXXInfo.SerialNumber[i], end='')
        print("")
        print("--Function String: %s" % bytes(USB2XXXFunctionString.value).decode('ascii'))
    else:
        print("Get device infomation faild!")
        exit()
    # 初始化CAN
    CANConfig = CANFD_INIT_CONFIG()
    # 获取波特率参数
    ret = CANFD_GetCANSpeedArg(DevHandles[0], byref(CANConfig), 500000, 2000000)
    if (ret != CANFD_SUCCESS):
        print("Get CAN speed failed!")
        exit()
    else:
        print("Get CAN speed Success!")

    ret = CANFD_Init(DevHandles[0], CAN1, byref(CANConfig))
    if (ret != CANFD_SUCCESS):
        print("Config CAN1 failed!")
        exit()
    else:
        print("Config CAN1 Success!")
        ret = CANFD_Init(DevHandles[0], CAN2, byref(CANConfig))
    if (ret != CANFD_SUCCESS):
        print("Config CAN2 failed!")
        exit()
    else:
        print("Config CAN2 Success!")
    return DevHandles, CAN1, CAN2


# 用于发送CPD启停报文
def send_cmd_msg(DevHandles, CAN1, ID, byte3):
    CanMsg = CANFD_MSG()
    for i in range(0, 1):
        CanMsg.Flags = 0
        CanMsg.ID = ID
        CanMsg.DLC = 8  # 数据字节长度
        CanMsg.TimeStamp = 100  # 帧接收或者发送时的时间戳，单位为10us
        CanMsg.Data[0] = 0x00
        CanMsg.Data[1] = 0x02
        CanMsg.Data[2] = 0x00
        CanMsg.Data[3] = byte3
        CanMsg.Data[4] = 0x00
        CanMsg.Data[5] = 0x00
        CanMsg.Data[6] = 0x00
        CanMsg.Data[7] = 0x00
    SendMsgNum = 1
    re = CANFD_SendMsg(DevHandles[0], CAN1, CanMsg, SendMsgNum)
    print(" \n发送报文ID:%02X" % CanMsg.ID)
    for j in range(0, CanMsg.DLC):
        print("%02X " % CanMsg.Data[j], end='')
    print("\n---------")

    return re


# 用于发送NM报文
def send_nm_msg(DevHandles, CAN1, ID):
    CanMsg = CANFD_MSG()
    for i in range(0, 1):
        CanMsg.Flags = 0
        CanMsg.ID = ID
        CanMsg.DLC = 8  # 数据字节长度
        CanMsg.TimeStamp = 100  # 帧接收或者发送时的时间戳，单位为10us
        CanMsg.Data[0] = 0x00
        CanMsg.Data[1] = 0x40
        CanMsg.Data[2] = 0x04
        CanMsg.Data[3] = 0x00
        CanMsg.Data[4] = 0x00
        CanMsg.Data[5] = 0x00
        CanMsg.Data[6] = 0x00
        CanMsg.Data[7] = 0x00
    SendMsgNum = 1
    CANFD_SendMsg(DevHandles[0], CAN1, CanMsg, SendMsgNum)

# 用于发送CIR信号报文
def send_cir_msg(DevHandles, CAN1, ID, CirData):
    CanMsg = CANFD_MSG()
    CanMsg.Flags = 0
    CanMsg.ID = ID
    CanMsg.DLC = 64  # 数据字节长度
    CanMsg.TimeStamp = 100  # 帧接收或者发送时的时间戳，单位为10us
    if CirData:  # 如果 CirData 不是空列表
        hex_list = CirData[0].split()
        number_list = [int(x, 16) for x in hex_list]
        for i in range(0, 64):
            CanMsg.Data[i] = number_list[i]
        SendMsgNum = 1
        if CanMsg.Data[3] == 0x30:
            CanMsg.Data[0] = 0x3B
            CanMsg.Data[1] = 0x00
            CanMsg.Data[2] = 0x00
            CanMsg.Data[3] = 0x90
        elif CanMsg.Data[3] == 0x31:
            CanMsg.Data[0] = 0x3B
            CanMsg.Data[1] = 0x00
            CanMsg.Data[2] = 0x00
            CanMsg.Data[3] = 0x54
        elif CanMsg.Data[3] == 0x32:
            CanMsg.Data[0] = 0x2B
            CanMsg.Data[1] = 0x00
            CanMsg.Data[2] = 0x00
            CanMsg.Data[3] = 0x18
        re = CANFD_SendMsg(DevHandles[0], CAN1, CanMsg, SendMsgNum)
        return re
    else:
        print("错误：CirData 是空的！")
        return None

    # print(" \n发送报文ID:%02X" %CanMsg.ID)
    # for j in range(0, CanMsg.DLC):
    #     print("%02X "%CanMsg.Data[j], end='')
    # print("\n---------")

def send_msg(DevHandles, CAN_Chanl, ID, DLC, Data, FrameInterval, SendMsgNum):
    CanMsg = CANFD_MSG()
    for i in range(0, 1):
        CanMsg.Flags = 0
        CanMsg.ID = ID
        CanMsg.DLC = DLC  # 数据字节长度
        CanMsg.TimeStamp = FrameInterval  # 帧接收或者发送时的时间戳，单位为10us
        for i in range(0, DLC):
            CanMsg.Data[i] = Data[i]
    ret = CANFD_SendMsg(DevHandles[0], CAN_Chanl, CanMsg, SendMsgNum)
    return ret

# 用于读取报文
def read_msg(DevHandles, CAN1, CanMsgBuffer):
    CanNum = CANFD_GetMsg(DevHandles[0], CAN1, byref(CanMsgBuffer), 1024000)
    return CanNum
    # if CanNum > 0:
    #     for i in range(0, CanNum):
    #         #检出结果报文
    #         if CanMsgBuffer[i].ID == 0x495:
    #             print("报文ID:0x495")
    #             print(CanMsgBuffer[i].Data[0:63])
    #             return CanMsgBuffer[i].Data[0:63]
    #         #参数报文
    #         elif CanMsgBuffer[i].ID == 0x700:
    #
    #             byte5 = CanMsgBuffer[i].Data[5]
    #             byte6 = CanMsgBuffer[i].Data[6]
    #             byte7 = CanMsgBuffer[i].Data[7]
    #
    #             # 模拟测试
    #             # byte5 = random.randint(0, 2)
    #             # byte6 = random.randint(0, 10)
    #             # byte7 = CanMsgBuffer[i].Data[1]
    #
    #             para_name = PARA[byte6]
    #             if para_name == 'Tx_Power':
    #
    #                 int_value = int.from_bytes(byte7.to_bytes(1, byteorder='little'), byteorder='little', signed=True)
    #                 para_num = int_value
    #             else:
    #                 para_num = byte7 * NUM_TYPE[byte5]
    #
    #             return "para", para_name, para_num
    #         else:
    #             return None
    # else:
    #     return None
