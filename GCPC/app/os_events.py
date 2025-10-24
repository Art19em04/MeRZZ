# app/os_events.py
import sys, time
import ctypes
from ctypes import wintypes

# --- Windows only ---
IS_WIN = sys.platform.startswith("win")

if IS_WIN:
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    # В ctypes.wintypes нет ULONG_PTR → объявляем безопасно для x86/x64.
    if not hasattr(wintypes, "ULONG_PTR"):
        try:
            # частая практика: алиаснуть к WPARAM (pointer-sized unsigned)
            wintypes.ULONG_PTR = wintypes.WPARAM
        except Exception:
            # запасной путь по размеру указателя
            wintypes.ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    # Константы SendInput
    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1
    INPUT_HARDWARE = 2

    KEYEVENTF_KEYUP     = 0x0002
    KEYEVENTF_SCANCODE  = 0x0008
    KEYEVENTF_EXTENDEDKEY = 0x0001

    # Определения структур из winuser.h / MSDN (см. KEYBDINPUT)
    # https://learn.microsoft.com/windows/win32/api/winuser/ns-winuser-keybdinput
    class KEYBDINPUT(ctypes.Structure):
        _fields_ = (
            ("wVk",       wintypes.WORD),
            ("wScan",     wintypes.WORD),
            ("dwFlags",   wintypes.DWORD),
            ("time",      wintypes.DWORD),
            ("dwExtraInfo", wintypes.ULONG_PTR),
        )

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = (
            ("dx",        wintypes.LONG),
            ("dy",        wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags",   wintypes.DWORD),
            ("time",      wintypes.DWORD),
            ("dwExtraInfo", wintypes.ULONG_PTR),
        )

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = (
            ("uMsg",    wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        )

    class _INPUTunion(ctypes.Union):
        _fields_ = (
            ("ki", KEYBDINPUT),
            ("mi", MOUSEINPUT),
            ("hi", HARDWAREINPUT),
        )

    class INPUT(ctypes.Structure):
        _anonymous_ = ("_iu",)
        _fields_ = (
            ("type", wintypes.DWORD),
            ("_iu",  _INPUTunion),
        )

    LPINPUT = ctypes.POINTER(INPUT)

    # Прототип SendInput
    user32.SendInput.argtypes = (wintypes.UINT,  # nInputs
                                 LPINPUT,        # pInputs
                                 ctypes.c_int)   # cbSize
    user32.SendInput.restype  = wintypes.UINT

    # Простая карта имён → VK (для букв/цифр берём ord)
    VK = {
        "CTRL": 0x11, "SHIFT": 0x10, "ALT": 0x12, "WIN": 0x5B,
        "ENTER": 0x0D, "ESC": 0x1B, "TAB": 0x09, "SPACE": 0x20,
        "LEFT": 0x25, "UP": 0x26, "RIGHT": 0x27, "DOWN": 0x28,
        "C": 0x43, "V": 0x56, "X": 0x58, "Z": 0x5A,
    }

    def _vk_of(token: str) -> int:
        t = token.upper()
        if len(t) == 1:
            return ord(t)  # для A..Z, 0..9 VK совпадает с ASCII
        return VK.get(t, 0)

    def _send_vk(vk: int, keyup: bool=False):
        flags = KEYEVENTF_KEYUP if keyup else 0
        ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
        inp = INPUT(type=INPUT_KEYBOARD, ki=ki)
        n = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        if n != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def _press_combo(tokens):
        # модификаторы удерживаем, последнюю клавишу кликаем
        mods = [t for t in tokens[:-1]]
        main = tokens[-1]
        for m in mods:
            _send_vk(_vk_of(m), keyup=False)
        _send_vk(_vk_of(main), keyup=False)
        _send_vk(_vk_of(main), keyup=True)
        for m in reversed(mods):
            _send_vk(_vk_of(m), keyup=True)

    def send_keys(tokens):
        """
        tokens: список строк, напр. ["CTRL","C"] или ["CTRL","SHIFT","Z"]
        """
        if not tokens:
            return
        try:
            _press_combo(tokens)
        except Exception as e:
            # fallback: небольшая задержка и повтор
            time.sleep(0.01)
            _press_combo(tokens)

else:
    # Для не-Windows просто глушим (или реализуем X11/Wayland отдельно)
    def send_keys(tokens):
        return
