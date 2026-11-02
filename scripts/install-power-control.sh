#!/bin/sh
# 允许 Spark Dashboard 一键开/关桌面、重启、关机（只能执行 spark-ctl 里的固定动作）
set -e
sudo install -m755 "$(dirname "$0")/../deploy/bin/spark-ctl" /usr/local/bin/spark-ctl
echo "$(id -un) ALL=(root) NOPASSWD: /usr/local/bin/spark-ctl" | sudo tee /etc/sudoers.d/spark >/dev/null
sudo chmod 440 /etc/sudoers.d/spark
sudo visudo -c >/dev/null && echo "完成：面板上的桌面开关、重启、关机按钮已可用"
