import ctypes
import sys
from ctypes import wintypes

if not sys.platform.startswith("win"):
    raise RuntimeError("Windows platform required for send_keys")

user32 = ctypes.WinDLL("user32", use_last_error=True)
if not hasattr(wintypes, "ULONG_PTR"):
    wintypes.ULONG_PTR = wintypes.WPARAM if hasattr(wintypes, "WPARAM") else (
        ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
    )

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.ULONG_PTR),
    )


class _INPUTunion(ctypes.Union):
    _fields_ = (("ki", KEYBDINPUT),)


class INPUT(ctypes.Structure):
    _anonymous_ = ("_iu",)
    _fields_ = (("type", wintypes.DWORD), ("_iu", _INPUTunion))


LPINPUT = ctypes.POINTER(INPUT)
user32.SendInput.argtypes = (wintypes.UINT, LPINPUT, ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

VK = {
    "CTRL": 0x11,
    "SHIFT": 0x10,
    "ALT": 0x12,
    "WIN": 0x5B,
    "ENTER": 0x0D,
    "ESC": 0x1B,
    "TAB": 0x09,
    "SPACE": 0x20,
    "LEFT": 0x25,
    "UP": 0x26,
    "RIGHT": 0x27,
    "DOWN": 0x28,
    "C": 0x43,
    "V": 0x56,
    "X": 0x58,
    "Z": 0x5A,
}


def _vk_of(token: str) -> int:
    t = token.upper()
    return ord(t) if len(t) == 1 else VK.get(t, 0)


def _send_vk(vk: int, keyup: bool = False):
    flags = KEYEVENTF_KEYUP if keyup else 0
    ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
    inp = INPUT(type=INPUT_KEYBOARD, ki=ki)
    n = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    if n != 1:
        raise ctypes.WinError(ctypes.get_last_error())


def _press_combo(tokens):
    mods = tokens[:-1]
    main = tokens[-1]
    for m in mods:
        _send_vk(_vk_of(m), keyup=False)
    _send_vk(_vk_of(main), keyup=False)
    _send_vk(_vk_of(main), keyup=True)
    for m in reversed(mods):
        _send_vk(_vk_of(m), keyup=True)


def send_keys(tokens):
    if not tokens:
        return
    _press_combo(tokens)
