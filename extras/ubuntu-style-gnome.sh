#!/usr/bin/env bash
# 在 Omarchy (Arch) 上安装 GNOME，并配置成 Ubuntu 风格桌面。
# Hyprland/Omarchy 会保留，可在登录界面切换回去。
set -euo pipefail

if [[ $EUID -eq 0 ]]; then
  echo "请用普通用户运行（脚本会在需要时调用 sudo）"; exit 1
fi

echo "==> 1/4 安装 GNOME 桌面和 Ubuntu 主题/字体"
sudo pacman -S --needed --noconfirm \
  gnome-shell gnome-session gnome-control-center gnome-tweaks \
  gnome-shell-extensions gnome-shell-extension-appindicator \
  gnome-console gnome-text-editor nautilus xdg-desktop-portal-gnome \
  gnome-keyring gnome-backgrounds \
  yaru-icon-theme yaru-gnome-shell-theme yaru-sound-theme \
  ttf-ubuntu-font-family

echo "==> 2/4 安装 Ubuntu 风格左侧 Dock (dash-to-dock, AUR)"
sudo pacman -S --needed --noconfirm --assume-installed gtk-engine-murrine yaru-gtk-theme  # murrine 仅用于 GTK2，ARM 上编译不了
yay -S --needed --noconfirm gnome-shell-extension-dash-to-dock

echo "==> 3/4 写入 Ubuntu 风格设置"
EXT=/usr/share/gnome-shell/extensions
gs() {  # gs <扩展目录名|-> <schema> <key> <value>
  local dir="$1"; shift
  if [[ "$dir" != "-" && -d "$EXT/$dir/schemas" ]]; then
    gsettings --schemadir "$EXT/$dir/schemas" set "$@"
  else
    gsettings set "$@"
  fi
}

gs - org.gnome.shell disable-user-extensions false
gs - org.gnome.shell enabled-extensions \
  "['dash-to-dock@micxgx.gmail.com', 'appindicatorsupport@rgcjonas.gmail.com', 'user-theme@gnome-shell-extensions.gcampax.github.com']"

gs - org.gnome.desktop.interface gtk-theme 'Yaru'
gs - org.gnome.desktop.interface icon-theme 'Yaru'
gs - org.gnome.desktop.interface cursor-theme 'Yaru'
gs - org.gnome.desktop.interface font-name 'Ubuntu 11'
gs - org.gnome.desktop.interface document-font-name 'Ubuntu 11'
gs - org.gnome.desktop.interface monospace-font-name 'Ubuntu Mono 13'
gs - org.gnome.desktop.wm.preferences titlebar-font 'Ubuntu Bold 11'
gs - org.gnome.desktop.wm.preferences button-layout ':minimize,maximize,close'
gs - org.gnome.desktop.sound theme-name 'Yaru'
gs user-theme@gnome-shell-extensions.gcampax.github.com \
  org.gnome.shell.extensions.user-theme name 'Yaru'

D=dash-to-dock@micxgx.gmail.com
gs $D org.gnome.shell.extensions.dash-to-dock dock-position 'LEFT'
gs $D org.gnome.shell.extensions.dash-to-dock extend-height true
gs $D org.gnome.shell.extensions.dash-to-dock dock-fixed true
gs $D org.gnome.shell.extensions.dash-to-dock dash-max-icon-size 48
gs $D org.gnome.shell.extensions.dash-to-dock show-apps-at-top true
gs $D org.gnome.shell.extensions.dash-to-dock show-mounts true
gs $D org.gnome.shell.extensions.dash-to-dock click-action 'focus-minimize-or-previews'
gs $D org.gnome.shell.extensions.dash-to-dock custom-theme-shrink true

echo "==> 4/4 登录时自动进入 GNOME（原来是 Omarchy）"
sudo sed -i 's/^Session=.*/Session=gnome.desktop/' /etc/sddm.conf.d/*.conf

echo
echo "完成！重启电脑（或注销）后即进入 Ubuntu 风格的 GNOME 桌面。"
echo "想切回 Omarchy：sudo sed -i 's/^Session=.*/Session=omarchy.desktop/' /etc/sddm.conf.d/*.conf"
