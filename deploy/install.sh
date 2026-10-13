#!/bin/sh
# Install gpu-powerd on a host. Run as root from the directory holding the bundle.
#
#   sudo ./install.sh              # install the binary, unit and (first time) config; start it
#   sudo ./install.sh --uninstall  # stop it, restore what it replaced
#
# The boot-time nvidia-power-limit.service (if present) is disabled, and that is
# recorded in /var/lib/gpu-powerd/replaced so --uninstall can re-enable it.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
sbin=/usr/local/sbin/gpu-powerd
unit=/etc/systemd/system/gpu-powerd.service
config_dir=/etc/gpu-powerd
config=$config_dir/config.toml
state_dir=/var/lib/gpu-powerd
record=$state_dir/replaced
replaced_unit=nvidia-power-limit.service

if [ "$(id -u)" -ne 0 ]; then
    echo "install.sh must run as root" >&2
    exit 1
fi

if [ "${1:-}" = "--uninstall" ]; then
    # Stopping runs ExecStopPost (--restore): every GPU goes to its fallback_w.
    systemctl disable --now gpu-powerd 2>/dev/null || true
    rm -f "$unit"
    systemctl daemon-reload
    if [ -f "$record" ]; then
        while read -r name; do
            [ -n "$name" ] || continue
            if systemctl cat "$name" >/dev/null 2>&1; then
                systemctl enable "$name"
                systemctl start "$name" || echo "warning: $name did not start" >&2
                echo "Re-enabled $name."
            fi
        done < "$record"
        rm -f "$record"
    fi
    rm -f "$sbin"
    echo "Removed gpu-powerd. $config_dir and $state_dir (config, learned model) were left in place."
    exit 0
fi

for f in gpu-powerd gpu-powerd.service config.example.toml; do
    [ -f "$here/$f" ] || { echo "missing $here/$f" >&2; exit 1; }
done

# Refuse a host without NVIDIA GPUs (the binary's own read-only check, via NVML).
if ! "$here/gpu-powerd" --check --config /dev/null >/dev/null; then
    echo "error: no usable NVIDIA GPUs on this host (gpu-powerd --check failed); nothing installed" >&2
    exit 1
fi

install -d -o root -g root -m 0755 "$config_dir" "$state_dir"
# Validate an existing config against this binary before replacing anything.
if [ -f "$config" ] && ! "$here/gpu-powerd" --check --config "$config" >/dev/null; then
    echo "error: $config does not validate with the new binary; fix it first (nothing installed)" >&2
    exit 1
fi

install -o root -g root -m 0755 "$here/gpu-powerd" "$sbin"
install -o root -g root -m 0644 "$here/gpu-powerd.service" "$unit"
install -o root -g root -m 0644 "$here/config.example.toml" "$config_dir/config.example.toml"
if [ ! -f "$config" ]; then
    install -o root -g root -m 0644 "$here/config.example.toml" "$config"
    echo "Wrote $config from the example; edit it, then: systemctl reload gpu-powerd"
fi

# Replace the boot-time fixed limit, keeping a record for --uninstall.
if systemctl is-enabled --quiet "$replaced_unit" 2>/dev/null; then
    grep -qx "$replaced_unit" "$record" 2>/dev/null || echo "$replaced_unit" >> "$record"
    systemctl disable "$replaced_unit"
    echo "Disabled $replaced_unit (recorded in $record; --uninstall re-enables it)."
fi

systemctl daemon-reload
systemctl enable gpu-powerd
systemctl restart gpu-powerd
echo "Installed $sbin and $unit."
