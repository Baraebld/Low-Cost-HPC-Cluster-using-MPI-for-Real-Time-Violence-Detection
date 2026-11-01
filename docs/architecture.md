# System Architecture

This document describes the heterogeneous distributed architecture of the **Low-Cost HPC Cluster using MPI for Real-Time Violence Detection**, detailing the interaction between the physical hardware, networking fabric, MPI communication layer, deep learning pipeline, and user demonstration interface.

---

## 1. High-Level Architecture Overview

The system is designed as a **heterogeneous master-worker cluster** comprising a desktop workstation (Master PC) and four diskless **Raspberry Pi 4 Model B** single-board computers (Compute Nodes), interconnected via an Ethernet switch.

```mermaid
graph TB
    subgraph Client Layer
        Browser["🌐 Web Browser / End User"]
    end

    subgraph Master Node ["🖥️ Master Host (PC) - 192.168.50.1"]
        UI["Streamlit Web App (Port 8501)"]
        Preproc["OpenCV Video Preprocessing<br/>Uniform Sampling & Normalization"]
        MPI_Master["MPI Rank 0 (Master Process)<br/>Workload Coordinator & Aggregator"]
        NFS_Srv["NFS Kernel Server<br/>Exports /srv/nfs/slave*"]
        DNSMASQ["dnsmasq<br/>DHCP + TFTP Boot Service"]
        NAT["iptables NAT Gateway<br/>Internet Sharing"]
    end

    subgraph Interconnect Fabric ["🔌 Network Fabric (192.168.50.0/24)"]
        Switch["TP-Link TL-SF1008D 100Mbps Switch<br/>(12.5 MB/s Throughput Limit)"]
    end

    subgraph Worker Nodes ["🍓 Diskless Raspberry Pi 4 Cluster"]
        direction TB
        Pi1["Pi Node 1 (Rank 1)<br/>192.168.50.101<br/>TorchScript / LibTorch"]
        Pi2["Pi Node 2 (Rank 2)<br/>192.168.50.102<br/>TorchScript / LibTorch"]
        Pi3["Pi Node 3 (Rank 3)<br/>192.168.50.103<br/>TorchScript / LibTorch"]
        Pi4["Pi Node 4 (Rank 4)<br/>192.168.50.104<br/>TorchScript / LibTorch"]
    end

    Browser <-->|HTTP / Video Stream| UI
    UI --> Preproc
    Preproc --> MPI_Master
    MPI_Master <-->|OpenMPI TCP btl| Switch
    Switch <-->|Cat6 Ethernet| Pi1
    Switch <-->|Cat6 Ethernet| Pi2
    Switch <-->|Cat6 Ethernet| Pi3
    Switch <-->|Cat6 Ethernet| Pi4
    DNSMASQ -.->|PXE / TFTP Boot| Switch
    NFS_Srv -.->|Root Filesystem Mount| Switch
```

---

## 2. Hardware Specifications

| Component | Master Node (PC) | Worker Nodes (4x Raspberry Pi 4B) |
|---|---|---|
| **CPU** | x86_64 Multi-core Host Processor | Quad-core ARM Cortex-A72 (ARMv8) @ 1.5 GHz |
| **RAM** | 16 GB DDR4 | 4 GB / 8 GB LPDDR4-3200 per node (16–32 GB total) |
| **Storage** | NVMe SSD (Hosts golden root filesystems) | Diskless (Diskless PXE netboot over NFS) |
| **Network Interface** | Dual NICs (Uplink + Cluster interface) | Integrated Gigabit Ethernet (Broadcom BCM54213PE) |
| **Role** | Gateway, NFS/TFTP Server, MPI Master (Rank 0) | Dedicated MPI Compute Workers (Ranks 1–4) |
| **Operating System** | Ubuntu Linux / Debian x86_64 | Raspberry Pi OS (Debian 12 Bookworm, 64-bit) |

### Network Fabric
- **Switch**: TP-Link TL-SF1008D 8-Port Fast Ethernet Switch (100 Mbps).
- **Physical Cabling**: High-grade Category 6 (Cat6) UTP Ethernet cables.
- **Subnet Configuration**: Private subnet `192.168.50.0/24`.
  - Master Host: `192.168.50.1` (Gateway & DNS/DHCP server)
  - Worker 1: `192.168.50.101`
  - Worker 2: `192.168.50.102`
  - Worker 3: `192.168.50.103`
  - Worker 4: `192.168.50.104`

---

## 3. End-to-End Data Flow

The runtime data pipeline for distributed violence detection operates through five primary stages:

```mermaid
sequenceDiagram
    autonumber
    actor User as Web User
    participant Streamlit as Streamlit Frontend
    participant Master as Master PC (MPI Rank 0)
    participant Switch as 100Mbps Network
    participant Workers as 4x Raspberry Pis (Ranks 1..4)

    User->>Streamlit: Upload surveillance video (.mp4)
    Streamlit->>Master: Invoke video ingestion
    Master->>Master: OpenCV uniform sampling (16 frames)<br/>RGB conversion & ImageNet normalization
    Master->>Master: Assemble tensor [1, 16, 3, 224, 224]
    
    rect rgb(240, 248, 255)
        note over Master, Workers: MPI Distributed Inference Phase
        Master->>Switch: Broadcast tensor [1, 16, 3, 224, 224]
        Switch->>Workers: Deliver tensor to Ranks 1, 2, 3, 4
        Workers->>Workers: Execute TorchScript JIT forward pass<br/>(R3D-18 Backbone + Transformer)
        Workers->>Workers: Compute Sigmoid Probability & Uncertainty
        Workers->>Switch: Send (prob, uncertainty) via MPI_Send
        Switch->>Master: Deliver worker results via MPI_Recv
    end

    Master->>Master: Aggregate predictions (Max / Average)<br/>Compare against optimal threshold (0.65)
    Master->>Streamlit: Return violence probability & alert state
    Streamlit->>User: Display confidence score, video preview & decision banner
```

---

## 4. Key Architectural Decisions

1. **Diskless Architecture (PXE + NFS)**:
   - Eliminates micro-SD card wear and corruption across the cluster.
   - Centralizes maintenance: all OS updates, libraries, and model weights are updated once on the host's golden image (`/srv/nfs/mypi`).
2. **Heterogeneous Master-Worker Computing**:
   - Offloads video decoding, UI serving, and job coordination to the Master PC.
   - Dedicates all Raspberry Pi CPU cores to mathematical tensor inference.
3. **Preserving Full Temporal Sequences**:
   - Rather than fragmenting videos into micro-clips of 3–4 frames (which breaks long-range action dependencies), workers process full 16-frame sequences, allowing the Transformer temporal encoder to reason over the entire motion trajectory.
