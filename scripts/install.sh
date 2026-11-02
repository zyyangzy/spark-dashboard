#!/usr/bin/env bash
# Install Spark Dashboard for the current user.
# Tested only on the author's machine: Omarchy (Arch Linux, aarch64) on an HP OmniBook X 14 with an RTX Spark N1X.
# Re-running is safe: existing tokens and an existing ~/.config/spark/Caddyfile are kept.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
SPARK=$HOME/.local/share/spark
CONF=$HOME/.config/spark
UNITS=$HOME/.config/systemd/user

need() { command -v "$1" >/dev/null 2>&1 || echo "  - missing: $1 ($2)"; }
echo "==> Checking prerequisites"
need python3 "required"
need docker "required: containers, vLLM, TensorFold, Open WebUI"
need caddy "required: single entry point on port 11000"
need nvidia-smi "required: GPU monitoring"
need nvidia-ctk "required for GPU containers: nvidia-container-toolkit"
need ollama "optional: Ollama tab"
need tailscale "optional: Network tab"
need opencode "optional: opencode service"

echo "==> Installing the dashboard to $SPARK/dashboard"
mkdir -p "$SPARK/dashboard" "$CONF" "$UNITS" "$HOME/notebooks"
install -m644 "$REPO"/dashboard/*.py "$REPO"/dashboard/index.html "$SPARK/dashboard/"
[ -f "$CONF/Caddyfile" ] || install -m644 "$REPO/deploy/caddy/Caddyfile" "$CONF/Caddyfile"
for t in jupyter_token vscode_token opencode_token; do
  [ -f "$CONF/$t" ] || (umask 077; python3 -c "import secrets; print(secrets.token_urlsafe(24))" > "$CONF/$t")
done
install -m644 "$REPO"/deploy/systemd/user/*.service "$UNITS/"

echo "==> Installing system services (needs sudo)"
render() { sed -e "s#@USER@#$(id -un)#g" -e "s#@HOME@#$HOME#g" "$1"; }
render "$REPO/deploy/systemd/system/spark-dashboard.service.in" | sudo tee /etc/systemd/system/spark-dashboard.service >/dev/null
if grep -q 'Name="Lid Switch"' /proc/bus/input/devices 2>/dev/null; then
  # Laptop: lid closed → screen off (no sleep), plus the dashboard's screen on/off switch
  sudo install -m755 "$REPO/deploy/bin/lid-backlight.py" /usr/local/bin/lid-backlight.py
  render "$REPO/deploy/systemd/system/lid-backlight.service.in" | sudo tee /etc/systemd/system/lid-backlight.service >/dev/null
  LID=lid-backlight
fi
sudo systemctl daemon-reload
sudo systemctl enable --now spark-dashboard ${LID:-}
sudo loginctl enable-linger "$(id -un)"   # keep user services running without a login session

echo "==> Starting the entry point (http://localhost:11000)"
systemctl --user daemon-reload
systemctl --user enable --now spark-proxy

cat <<EOF

Done. Open http://localhost:11000 (over SSH: ssh -L 11000:localhost:11000 <host>).

Optional components (see README.md):
  JupyterLab   create ~/.local/share/spark/jupyter-venv, then: systemctl --user enable --now spark-jupyter
  VS Code web  put the VS Code CLI at ~/.local/share/spark/vscode-cli, then: systemctl --user enable --now spark-code
  opencode     systemctl --user enable --now spark-opencode
  Power menu   $REPO/scripts/install-power-control.sh   (desktop on/off, restart, shut down)
EOF
