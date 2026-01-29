# -*- coding: utf-8 -*-
import ctypes
import time
from ctypes import wintypes

if not hasattr(wintypes, "ULONG_PTR"): wintypes.ULONG_PTR = wintypes.WPARAM
ULONG_PTR = wintypes.ULONG_PTR;
DWORD = wintypes.DWORD
WORD = wintypes.WORD
LONG = wintypes.LONG
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MAPVK_VK_TO_VSC = 0


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [('wVk', WORD), ('wScan', WORD), ('dwFlags', DWORD), ('time', DWORD), ('dwExtraInfo', ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [('dx', LONG), ('dy', LONG), ('mouseData', DWORD), ('dwFlags', DWORD), ('time', DWORD),
                ('dwExtraInfo', ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [('uMsg', DWORD), ('wParamL', WORD), ('wParamH', WORD)]


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [('ki', KEYBDINPUT), ('mi', MOUSEINPUT), ('hi', HARDWAREINPUT)]

    _anonymous_ = ('ii',)
    _fields_ = [('type', DWORD), ('ii', _INPUT)]


SendInput = ctypes.windll.user32.SendInput;
MapVirtualKey = ctypes.windll.user32.MapVirtualKeyW
VK = {'BACK': 0x08, 'TAB': 0x09, 'ENTER': 0x0D, 'SHIFT': 0x10, 'CTRL': 0x11, 'ALT': 0x12, 'PAUSE': 0x13, 'CAPS': 0x14,
      'ESC': 0x1B, 'SPACE': 0x20, 'PGUP': 0x21, 'PGDN': 0x22, 'END': 0x23, 'HOME': 0x24, 'LEFT': 0x25, 'UP': 0x26,
      'RIGHT': 0x27, 'DOWN': 0x28,
      'INS': 0x2D, 'DEL': 0x2E, '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34, '5': 0x35, '6': 0x36, '7': 0x37,
      '8': 0x38, '9': 0x39,
      'A': 0x41, 'B': 0x42, 'C': 0x43, 'D': 0x44, 'E': 0x45, 'F': 0x46, 'G': 0x47, 'H': 0x48, 'I': 0x49, 'J': 0x4A,
      'K': 0x4B, 'L': 0x4C, 'M': 0x4D, 'N': 0x4E, 'O': 0x4F, 'P': 0x50, 'Q': 0x51, 'R': 0x52, 'S': 0x53, 'T': 0x54,
      'U': 0x55, 'V': 0x56, 'W': 0x57, 'X': 0x58, 'Y': 0x59, 'Z': 0x5A,
      'LWIN': 0x5B, 'RWIN': 0x5C, 'APPS': 0x5D, 'F1': 0x70, 'F2': 0x71, 'F3': 0x72, 'F4': 0x73, 'F5': 0x74, 'F6': 0x75,
      'F7': 0x76, 'F8': 0x77, 'F9': 0x78, 'F10': 0x79, 'F11': 0x7A, 'F12': 0x7B}
EXTENDED_VK = set(
    [VK.get(k, 0) for k in ['INS', 'DEL', 'HOME', 'END', 'PGUP', 'PGDN', 'LEFT', 'RIGHT', 'UP', 'DOWN'] if k in VK])


def _key_event(vk, is_down=True, use_scan=False):
    """Construct a keyboard INPUT structure for a single key press/release."""
    scan = MapVirtualKey(vk, MAPVK_VK_TO_VSC)
    flags = (KEYEVENTF_SCANCODE if use_scan else 0) | (0 if is_down else KEYEVENTF_KEYUP)
    if vk in EXTENDED_VK: flags |= KEYEVENTF_EXTENDEDKEY
    ki = KEYBDINPUT(0 if use_scan else vk, 0 if not use_scan else scan, flags, 0, 0)
    return INPUT(type=INPUT_KEYBOARD, ki=ki)


def _mouse_input(dx=0, dy=0, data=0, flags=0):
    """Construct a mouse INPUT structure."""
    return INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dx, dy, data, flags, 0, 0))


def _send_inputs(ins, delay=0.02):
    """Send a batch of INPUT structures with optional delay."""
    if not ins:
        return
    arr = (INPUT * len(ins))(*ins)
    SendInput(len(ins), arr, ctypes.sizeof(INPUT))
    if delay:
        time.sleep(delay)


def _parse_combo(combo):
    """Parse human readable combo like CTRL+ALT+DEL into key codes."""
    parts = [p.strip().upper() for p in combo.split('+') if p.strip()]
    mods = []
    main = None
    for p in parts:
        if p in ('CTRL', 'SHIFT', 'ALT', 'LCTRL', 'RCTRL', 'LSHIFT', 'RSHIFT', 'LALT', 'RALT', 'LWIN', 'RWIN', 'WIN'):
            mods.append(p)
        else:
            main = p
    mods = [('LWIN' if m == 'WIN' else m) for m in mods]

    def _vk(name):
        if name and name.startswith('VK_'): name = name[3:]
        if name in VK: return VK[name]
        if name and len(name) == 1 and name.isalnum(): return VK[name]
        raise ValueError(f'Unknown key: {name}')

    main_vk = _vk(main) if main else None
    mod_vks = [_vk(m) for m in mods]
    return mod_vks, main_vk


def press_combo(combo, prefer_scan=False):
    """Press a keyboard shortcut combination on Windows."""
    mod_vks, main_vk = _parse_combo(combo)
    evts = []
    for vk in mod_vks: evts.append(_key_event(vk, True, prefer_scan))
    if main_vk is not None:
        evts.append(_key_event(main_vk, True, prefer_scan))
        evts.append(_key_event(main_vk, False, prefer_scan))
    for vk in reversed(mod_vks): evts.append(_key_event(vk, False, prefer_scan))
    _send_inputs(evts)


def _norm_coord(v):
    """Clamp normalized coordinate to 0..65535 range for absolute mouse move."""
    return max(0, min(65535, int(round(v * 65535.0))))


def mouse_move_normalized(x, y):
    """Move mouse using normalized coordinates spanning the virtual desktop."""
    xi = _norm_coord(x)
    yi = _norm_coord(y)
    flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    _send_inputs([_mouse_input(dx=xi, dy=yi, flags=flags)], delay=0.0)


def mouse_press(button="left"):
    """Press specified mouse button (left/right)."""
    btn = button.lower()
    if btn == "left":
        flags = MOUSEEVENTF_LEFTDOWN
    elif btn == "right":
        flags = MOUSEEVENTF_RIGHTDOWN
    else:
        raise ValueError(f"Unsupported mouse button: {button}")
    _send_inputs([_mouse_input(flags=flags)], delay=0.0)


def mouse_release(button="left"):
    """Release specified mouse button (left/right)."""
    btn = button.lower()
    if btn == "left":
        flags = MOUSEEVENTF_LEFTUP
    elif btn == "right":
        flags = MOUSEEVENTF_RIGHTUP
    else:
        raise ValueError(f"Unsupported mouse button: {button}")
    _send_inputs([_mouse_input(flags=flags)], delay=0.0)


def mouse_scroll(delta):
    """Scroll mouse wheel by a signed delta (positive=up, negative=down)."""
    _send_inputs([_mouse_input(data=int(delta), flags=MOUSEEVENTF_WHEEL)], delay=0.0)
