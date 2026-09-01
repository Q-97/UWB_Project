from app.comm.canfd import *
from app.comm.zlgcan import *
class DevicePara:
    ZCAN_DEVICE_TYPE = None
    CAN1 = None
    CAN2 = None
    zcanlib = None
    chn_handle = None
    DevHandles = None
    device_type = None
    CanMsgBuffer = (CANFD_MSG * 1024000)()

def init_device(deviceTypeselect):
    if deviceTypeselect == "ZLGCAN":
        params = zlgcan_init()
        if None in params:
            return "设备初始化失败！"
        else:
            DevicePara.ZCAN_DEVICE_TYPE, DevicePara.zcanlib, DevicePara.chn_handle = params
            DevicePara.device_type = "ZLG"
            return "已连接ZLG设备"
    elif deviceTypeselect == "TOOMOSS":
        DevicePara.device_type = "TUMOSI"
        tumosi_params = canfd_init()  # 你需要实现这个函数
        if tumosi_params is not None:
            DevicePara.DevHandles, DevicePara.CAN1, DevicePara.CAN2 = tumosi_params
            return "已连接图莫斯设备"
        else:
            return "设备初始化失败！"
    else:
        return "请选择设备类型"

def deinit_device(deviceTypeselect):
    if deviceTypeselect == "ZLGCAN":
        ret = DevicePara.zcanlib.CloseDevice(DevicePara.ZCAN_DEVICE_TYPE)
        if ret == 1:
            return "CloseDevice success! "
    elif deviceTypeselect == "TOOMOSS":
        ret = USB_CloseDevice(DevicePara.DevHandles)
        if (bool(ret)):
            return "CloseDevice success! "
    else:
        return "请选择设备类型"

def CAN_SendMsg(canID, dlc, data, sendMsgNum, frameInterval):
    if DevicePara.device_type == "ZLG":
        if DevicePara.chn_handle is not None and DevicePara.zcanlib is not None:
            zlgcan_send(DevicePara.chn_handle, canID, DevicePara.zcanlib, dlc, data, frameInterval, sendMsgNum)
    elif DevicePara.device_type == "TUMOSI":
        if DevicePara.DevHandles is not None and DevicePara.CAN1 is not None:
            send_msg(DevicePara.DevHandles, DevicePara.CAN1, canID, dlc, data, frameInterval, sendMsgNum)
    else:
        pass
    if canID not in [0x781, 0x500]:
        print(" \n发送报文ID:%02X" % canID)
        for j in range(0, dlc):
            print("%02X " % data[j], end='')
        print("\n---------")

def CAN_GetMsg():
    if DevicePara.device_type == "ZLG":
        num, msgs = ZLGread_msg(DevicePara.chn_handle, DevicePara.zcanlib)
        if num > 0:
            # for i in range(0, num):
            for i, msg in zip(range(num), msgs):
                if msg["type"] == "CAN":
                    DevicePara.CanMsgBuffer[i].ID = msg["can_id"]
                    DevicePara.CanMsgBuffer[i].DLC = msg["dlc"]
                    for j in range(0, msg["dlc"]):
                        DevicePara.CanMsgBuffer[i].Data[j] = msg["data"][j]
                else:
                    DevicePara.CanMsgBuffer[i].ID = msg["can_id"]
                    DevicePara.CanMsgBuffer[i].DLC = msg["len"]
                    for j in range(0, msg["len"]):
                        DevicePara.CanMsgBuffer[i].Data[j] = msg["data"][j]
        return num
    elif DevicePara.device_type == "TUMOSI":
        CanNum = read_msg(DevicePara.DevHandles, DevicePara.CAN1, DevicePara.CanMsgBuffer)
        return CanNum
    else:
        pass