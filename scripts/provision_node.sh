#!/usr/bin/env bash
# ==============================================================================
# provision_node.sh
# Provision a diskless Raspberry Pi node from the golden NFS image
# Referenced from Section 2.6.1 (Listing 13) of Project Report
# ==============================================================================

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <HOSTNAME> <IP> <MAC_ADDRESS> <SERIAL>"
    echo "Example: $0 slave1 192.168.50.101 dc:a6:32:xx:xx:xx 12345678"
    exit 1
fi

HOSTNAME="$1"
IP="$2"
MAC="$3"
SERIAL="$4"

GOLDEN_IMAGE="/srv/nfs/mypi"
SERVER_IP="192.168.50.1"

echo "=== Provisioning $HOSTNAME ($SERIAL) at $IP ==="

# 1. Create target directories and clone golden root filesystem
echo "[1/7] Cloning golden image to /srv/nfs/$HOSTNAME..."
sudo mkdir -p "/srv/nfs/$HOSTNAME"
sudo mkdir -p "/srv/tftpboot/$SERIAL"
sudo cp -a "$GOLDEN_IMAGE/." "/srv/nfs/$HOSTNAME/"

# 2. Set the node hostname
echo "[2/7] Setting node hostname..."
echo "$HOSTNAME" | sudo tee "/srv/nfs/$HOSTNAME/etc/hostname" > /dev/null

# 3. Write kernel cmdline for NFS root mount
echo "[3/7] Configuring kernel cmdline.txt..."
echo "console=serial0,115200 console=tty1 root=/dev/nfs nfsroot=${SERVER_IP}:/srv/nfs/${HOSTNAME},vers=3 rw ip=dhcp rootwait" \
    | sudo tee "/srv/nfs/$HOSTNAME/boot/cmdline.txt" > /dev/null

# 4. Bind-mount boot partition into TFTP tree
echo "[4/7] Updating /etc/fstab and mounting TFTP boot dir..."
FSTAB_LINE="/srv/nfs/$HOSTNAME/boot /srv/tftpboot/$SERIAL none defaults,bind 0 0"
if ! grep -q "/srv/tftpboot/$SERIAL" /etc/fstab; then
    echo "$FSTAB_LINE" | sudo tee -a /etc/fstab > /dev/null
fi
sudo mount "/srv/tftpboot/$SERIAL"

# 5. Register node in dnsmasq (Static DHCP assignment)
echo "[5/7] Registering static DHCP lease in /etc/dnsmasq.conf..."
DNSMASQ_ENTRY="dhcp-host=$MAC,$IP,$HOSTNAME"
if ! grep -q "$MAC" /etc/dnsmasq.conf; then
    echo "$DNSMASQ_ENTRY" | sudo tee -a /etc/dnsmasq.conf > /dev/null
fi

# 6. Add NFS export for this node
echo "[6/7] Adding NFS export in /etc/exports..."
NFS_ENTRY="/srv/nfs/$HOSTNAME $IP(rw,sync,no_subtree_check,no_root_squash)"
if ! grep -q "/srv/nfs/$HOSTNAME" /etc/exports; then
    echo "$NFS_ENTRY" | sudo tee -a /etc/exports > /dev/null
fi

# 7. Enable SSH on first boot
echo "[7/7] Enabling SSH service on node..."
sudo touch "/srv/nfs/$HOSTNAME/boot/ssh"

# Reload daemon, export NFS shares and restart dnsmasq
echo "Reloading services..."
sudo systemctl daemon-reload
sudo exportfs -ra
sudo systemctl restart dnsmasq

echo "=== Successfully provisioned $HOSTNAME! Node is ready to PXE netboot. ==="
