#!/usr/bin/env python3
# 屏幕控制：合盖关闭屏幕，开盖恢复（不睡眠）；
# 另外读取面板的开关文件 SCREEN_FLAG，存在时开盖也保持关闭（面板「关闭屏幕」）。
# OLED 没有背光，亮度调到 0 仍会发光，所以同时关闭显示输出（fbdev blank → DPMS Off），才是真正黑屏。
import fcntl, glob, os, re, select, struct

def find_lid():
    """在 /proc/bus/input/devices 里找名为 "Lid Switch" 的输入设备（编号因机器而异）。"""
    for block in open("/proc/bus/input/devices").read().split("\n\n"):
        if 'Name="Lid Switch"' in block:
            m = re.search(r"\bevent\d+", block)  # 形如 "H: Handlers=event0"
            if m:
                return f"/dev/input/{m.group(0)}"
    raise SystemExit("no Lid Switch input device found")


DEV = find_lid()
SCREEN_FLAG = os.environ.get("SPARK_SCREEN_FLAG", "/var/lib/spark/screen_off")  # 由 systemd 服务文件设置
EV_SW, SW_LID = 5, 0
EVIOCGSW = (2 << 30) | (8 << 16) | (ord("E") << 8) | 0x1B
BL = (glob.glob("/sys/class/backlight/*") or [None])[0]
FB_BLANK = "/sys/class/graphics/fb0/blank"           # 命令行控制台（桌面关闭时）可用
PANEL_DPMS = (glob.glob("/sys/class/drm/card*-eDP-*/dpms") or [None])[0]
saved = None


def write(name, value):
    try:
        with open(f"{BL}/{name}", "w") as f:
            f.write(str(value))
    except OSError:
        pass


def read(name):
    with open(f"{BL}/{name}") as f:
        return int(f.read())


def panel_on():
    try:
        with open(PANEL_DPMS) as f:
            return f.read().strip() == "On"
    except (OSError, TypeError):
        return None


def blank(on):
    try:
        with open(FB_BLANK, "w") as f:
            f.write("4" if on else "0")  # 4 = FB_BLANK_POWERDOWN
    except OSError:
        pass


def apply(off):
    global saved
    if off and panel_on():  # 按键等可能让控制台重新点亮，关屏期间重新关掉
        blank(True)
    if BL is None:
        return
    if off:
        if saved is None:
            saved = read("brightness") or read("max_brightness")
        if read("brightness") != 0:  # 期间若有人（如 GNOME）恢复了亮度，再关掉
            write("brightness", 0)
        write("bl_power", 4)
    elif saved is not None:
        blank(False)
        write("bl_power", 0)
        write("brightness", saved)
        saved = None


with open(DEV, "rb") as f:
    state = bytearray(8)
    fcntl.ioctl(f, EVIOCGSW, state)
    lid_closed = bool(state[0] & 1)
    while True:
        apply(lid_closed or os.path.exists(SCREEN_FLAG))
        if select.select([f], [], [], 1.0)[0]:
            _, _, typ, code, value = struct.unpack("llHHi", f.read(24))
            if typ == EV_SW and code == SW_LID:
                lid_closed = value == 1
