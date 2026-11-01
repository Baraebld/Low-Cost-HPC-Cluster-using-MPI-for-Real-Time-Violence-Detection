# Installation & Quick Start Guide

This guide provides instructions to run the violence detection demo locally on a single machine or reproduce the distributed MPI cluster pipeline across multiple nodes.

---

## 1. Mode 1: Local Demonstration (Single Machine)

You do **not** need the physical Raspberry Pi cluster to test the deep learning model or run the Streamlit web application.

### Prerequisites
- Python 3.10+ (Tested on Python 3.10 and 3.14)
- Git
- Web browser

### 1.1 Clone the Repository
```bash
git clone https://github.com/Baraebld/hpc-cluster-violence-detection.git
cd hpc-cluster-violence-detection
```

### 1.2 Set Up Virtual Environment
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 1.3 Install Dependencies
```bash
pip install -r requirements.txt
```

### 1.4 Download Model Checkpoint
Download the pre-trained weights (`hybrid_model3.pth`, ~162.7 MB) into the `models/` directory:
```bash
python scripts/download_model.py
```
*Alternatively, download `hybrid_model3.pth` manually from [Hugging Face Spaces](https://huggingface.co/spaces/Baraebld/Violence-Detection-model/tree/main/models) and place it under `models/`.*

### 1.5 Run Unit Tests
```bash
python -m pytest tests/
```

### 1.6 Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`. Upload any sample surveillance clip (`.mp4`, `.avi`, `.mov`) to view real-time predictions and confidence scores.

---

## 2. Mode 2: Distributed Cluster Execution (OpenMPI)

This mode runs the heterogeneous parallel inference pipeline across 1 Master PC and 4 Raspberry Pi nodes.

### Prerequisites
- 1 Master PC (Linux / Ubuntu) + 4 Raspberry Pi 4 Model B nodes.
- Fast Ethernet Switch or Gigabit Switch.
- OpenMPI 4.1.6 installed across all nodes:
  ```bash
  sudo apt update && sudo apt install -y openmpi-bin libopenmpi-dev libopencv-dev
  ```
- Passwordless Ed25519 SSH configured between Master and all Pis.

### 2.1 Set Up Network & SSH
On the Master PC:
```bash
# 1. Enable NAT masquerading and packet routing
sudo ./scripts/setup_network.sh

# 2. Configure passwordless Ed25519 SSH across 192.168.50.101..104
./scripts/setup_ssh_keys.sh
```

### 2.2 Compile MPI Binaries
From the `mpi/` directory on the Master host:
```bash
cd mpi
make benchmarks
```
This builds `hello_mpi`, `mpi_trap_pi`, and `matmul_benchmark`.

To compile the C++ LibTorch inference node:
```bash
make violence_mpi_master LIBTORCH_PATH=/usr/local/libtorch
```

### 2.3 Run Matrix Multiplication Benchmark
To reproduce the cluster benchmark (Table 1 in experimental analysis):
```bash
# Define cluster hosts in a hostfile
cat << EOF > hosts
192.168.50.101 slots=1
192.168.50.102 slots=1
192.168.50.103 slots=1
192.168.50.104 slots=1
EOF

# Execute 4-worker benchmark (N=2000)
mpirun --hostfile hosts -np 4 ./matmul_benchmark 2000
```

### 2.4 Run C++ LibTorch Distributed Violence Detection
```bash
# 1 Master PC + 4 Raspberry Pi workers (np = 5)
mpirun -np 5 --host 192.168.50.1,192.168.50.101,192.168.50.102,192.168.50.103,192.168.50.104 \
    ./violence_mpi_master ../models/hybrid_model3.pt ../data/surveillance_sample.mp4 0.65
```

### 2.5 Run Python `mpi4py` Distributed Video Inference
```bash
mpirun -np 5 --host 192.168.50.1,192.168.50.101,192.168.50.102,192.168.50.103,192.168.50.104 \
    python mpi/distributed_inference.py ../data/surveillance_sample.mp4
```

---

## 3. Common Troubleshooting

| Issue | Root Cause | Solution |
|---|---|---|
| `SSH Host Key Verification Failed` | Raspberry Pi re-flashed or re-imaged with new host keys | Run `ssh-keygen -f ~/.ssh/known_hosts -R 192.168.50.10X` |
| `btl_tcp_endpoint: connect failed` | MPI attempted to communicate over the wrong network interface | Add `--mca btl_tcp_if_include 192.168.50.0/24` to the `mpirun` command line |
| `FileNotFoundError: hybrid_model3.pth` | Model checkpoint not yet downloaded | Run `python scripts/download_model.py` |
| `OpenCV VideoCapture failed` | Video format codec missing on headless server | Install `ffmpeg` / `libavcodec-dev`: `sudo apt install ffmpeg` |
