# Evo-2 on AWS EC2 — Instance & Setup Guide

## Instance requirements
**GPU (must support FP8 path)**

* **Recommended** (single-GPU, cost-effective): G6e family (NVIDIA L40S, ~48 GB VRAM, Compute Capability 8.9/Ada).

  * Good sizes: ``g6e.4xlarge`` (1× L40S, 16 vCPU) or ``g6e.2xlarge`` (leaner CPU).

* **High-end / multi-GPU**: P5 family (NVIDIA H100, Compute Capability 9.0/Hopper).

Example: ``p5.48xlarge`` (8× H100) for large batch/multi-GPU or 40B model.

* Older G5/A10G (CC 8.6) are not ideal for Evo-2’s FP8 path. Start with ``evo2_7b`` on G6e/H100.

## AMI (OS image)

* **Deep Learning OSS NVIDIA Driver AMI – GPU PyTorch 2.8 (Ubuntu 24.04, x86_64)**
(Has recent NVIDIA driver, CUDA 12+, PyTorch, and prebuilt env.)

## Storage (EBS)

* **Root volume: 200 GB+ gp3** (recommended).

  * gp3 defaults are fine; if you stream a lot of weights, you can set ~3000 IOPS / 125 MB/s throughput.

* Why 200 GB? You’ll pull a large CUDA base image (if using Docker), install layers, and cache models.

## Networking / Security Group

Inbound: **SSH (22)** from your IP only.

Outbound: allow default (to pull from GitHub/Hugging Face).

Optional: expose **8888** (Jupyter) or other ports only if you need them

## Connect & sanity checks (on the instance)
```bash
# GPU and Python versions
nvidia-smi
python3 --version
```

* You should see NVIDIA L40S (G6e) or H100 (P5), and Python 3.12 on the DLAMI.

## (If needed) expand the root volume to 200 GB on a running instance

* After you “Modify volume” in the AWS console:
```bash
# See disks and filesystem
lsblk
df -Th
findmnt -n -o SOURCE,FSTYPE /

# Install growpart
sudo apt-get update
sudo apt-get install -y cloud-guest-utils

# Grow partition 1 on the first NVMe disk (adjust if your root differs)
sudo growpart /dev/nvme0n1 1

# Resize filesystem (ext4 vs xfs)
sudo resize2fs /dev/nvme0n1p1     # for ext4
# sudo xfs_growfs /                # use this instead if your root FS is xfs

# Verify space
df -h

```

## Choose your path

* Install Docker from Docker’s repo (avoids Ubuntu 24.04 containerd conflict)

```bash
# Remove conflicting bits (no error if not present)
sudo systemctl stop containerd || true
sudo apt-get remove -y containerd docker.io docker-doc podman-docker runc || true
sudo apt-get purge  -y containerd docker.io docker-doc podman-docker runc || true

# Install Docker CE (official repo for Ubuntu "noble")
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo $VERSION_CODENAME) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable Docker for your user
sudo usermod -aG docker $USER
newgrp docker
docker --version

```

* Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/ubuntu$(. /etc/os-release; echo $VERSION_ID)/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure
sudo systemctl restart docker

# Verify GPU visible inside containers
docker run --rm --gpus all ubuntu nvidia-smi

```

* Clone Evo-2, prepare cache, build image

```bash
# Hugging Face cache (persists across containers)
sudo mkdir -p /opt/hf-cache && sudo chown $USER:$USER /opt/hf-cache

# (optional) set your HF token to avoid rate limits
echo 'export HUGGINGFACE_HUB_TOKEN=hf_xxx' >> ~/.bashrc && source ~/.bashrc

git clone https://github.com/ArcInstitute/evo2
cd evo2
docker build -t evo2 .

```

If you previously ran out of space, reclaim and retry:

```bash
sudo systemctl stop docker
sudo rm -rf /var/lib/docker/buildkit
sudo systemctl start docker
docker system prune -a --volumes -f
```

* Run the container (with proper shared memory & ulimits)

```bash
docker run -it --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /opt/hf-cache:/root/.cache/huggingface \
  evo2 bash

```

(If ``--ipc=host`` isn’t allowed, use ``--shm-size=2g`` instead.)

* Smoke test (inside the container)

```bash
python -m evo2.test.test_evo2_generation --model_name evo2_7b
```

* Quick manual generation:

```bash
python - <<'PY'
from evo2 import Evo2
m = Evo2("evo2_7b")
out = m.generate(prompt_seqs=["ATGCGTATCG"], n_tokens=200, temperature=0.8, top_k=8)
print(out.sequences[0])
PY
```
