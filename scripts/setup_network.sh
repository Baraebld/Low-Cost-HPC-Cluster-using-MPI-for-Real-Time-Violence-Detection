#!/usr/bin/env bash
# ==============================================================================
# setup_network.sh
# Configure NAT masquerading and IP forwarding on Master PC Gateway (192.168.50.1)
# Referenced from Section 1.2.2 (Listing 1) of Project Report
# ==============================================================================

set -euo pipefail

# Network interfaces (Adjust interface names to match your hardware)
UPLINK_IF="${1:-enx022a010e1b03}"   # Master uplink interface connected to Internet
CLUSTER_IF="${2:-enx5c53102a71f8}"  # Master NIC connected to Cluster Switch (192.168.50.1)

echo "=== Configuring Cluster Gateway & NAT ==="
echo "Uplink interface : ${UPLINK_IF}"
echo "Cluster interface: ${CLUSTER_IF}"

# 1. Enable kernel IP forwarding
echo "Enabling kernel IPv4 forwarding..."
sudo sysctl -w net.ipv4.ip_forward=1

# Make IP forwarding persistent across reboots
if ! grep -q "net.ipv4.ip_forward=1" /etc/sysctl.conf; then
    echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf
fi

# 2. Configure iptables for NAT Masquerading
echo "Setting up iptables NAT rules..."
sudo iptables -t nat -A POSTROUTING -o "${UPLINK_IF}" -j MASQUERADE

# 3. Allow forwarding from cluster NIC to uplink NIC and established connections
sudo iptables -A FORWARD -i "${CLUSTER_IF}" -o "${UPLINK_IF}" -j ACCEPT
sudo iptables -A FORWARD -i "${UPLINK_IF}" -o "${CLUSTER_IF}" -m state --state RELATED,ESTABLISHED -j ACCEPT

echo "=== Network Gateway configuration complete ==="
