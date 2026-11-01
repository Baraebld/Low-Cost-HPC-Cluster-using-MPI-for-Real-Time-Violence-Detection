#!/usr/bin/env bash
# ==============================================================================
# setup_ssh_keys.sh
# Automated Ed25519 SSH Key Setup for Passwordless OpenMPI execution
# Referenced from Section 1.3 (Listings 2 & 3) of Project Report
# ==============================================================================

set -euo pipefail

SUBNET_PREFIX="192.168.50"
NODE_IPS=("100" "101" "102" "103" "104")
SSH_USER="pi"

echo "=== Setting up Ed25519 SSH Keys on Cluster Nodes ==="

# 1. Generate local key on Master if not present
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
    echo "Generating Master Ed25519 SSH key..."
    ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519"
fi

# 2. Generate Ed25519 keys on all Raspberry Pi nodes
for ip in "${NODE_IPS[@]}"; do
    TARGET_IP="${SUBNET_PREFIX}.${ip}"
    echo "Configuring SSH key on ${TARGET_IP}..."

    # Clear stale host keys if re-imaged
    ssh-keygen -f "$HOME/.ssh/known_hosts" -R "${TARGET_IP}" 2>/dev/null || true

    # Generate keypair on remote node
    ssh -o StrictHostKeyChecking=accept-new "${SSH_USER}@${TARGET_IP}" \
        "mkdir -p ~/.ssh && [ ! -f ~/.ssh/id_ed25519 ] && ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 || true"

    # Copy Master public key to node authorized_keys
    ssh-copy-id -i "$HOME/.ssh/id_ed25519.pub" -o StrictHostKeyChecking=accept-new "${SSH_USER}@${TARGET_IP}"
done

echo "=== Passwordless SSH setup complete across all nodes ==="
