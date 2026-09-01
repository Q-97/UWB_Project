"""语音播报封装（阶段 3：平台耦合收口）。

默认实现走 Windows PowerShell System.Speech（与原逻辑一致），
后续可替换为 pyttsx3 等跨平台实现而不影响调用方。
"""
import subprocess


class SpeechAnnouncer:
    """异步语音播报器（默认启用，可用 enabled=False 关闭）。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def speak_async(self, text: str) -> None:
        """异步播报文本；禁用或空文本时直接返回。"""
        if not text or not self.enabled:
            return
        escaped = text.replace("'", "''")
        cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{escaped}')"
        )
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[Speech] 播报失败: {e}")
