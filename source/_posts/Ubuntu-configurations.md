---
title: Ubuntu configurations
tags:
  - Ubuntu
category:
  - Ubuntu
date: 2025-01-20 11:17:07
---


## Install essential tools

```shell
apt update
apt install ubuntu-drivers-common
ubuntu-drivers autoinstall
apt install git tmux curl wget zsh unzip -y
apt install pkg-config libssl-dev build-essential -y
apt install python3-pip -y
```


## Set zsh as default

```shell
chsh -s $(which zsh)
echo $SHELL
```

## Install Oh-My-Zsh

```shell
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
sh -c "$(curl -fsSL https://install.ohmyz.sh/)"
```

## Install Miniconda

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
# After installation
source ~/miniconda3/bin/activate
conda init --all
```

## Install CUDA Toolkit 12.6

[Reference](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) 

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

## Install PyTorch v2.2.0

### Conda

OSX

```shell
# conda
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 -c pytorch
```

Linux and Windows

```shell
# CUDA 11.8
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
```

### Wheel

OSX

```shell
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
```

Linux and Windows

```shell
# ROCM 5.6 (Linux only)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.6
# CUDA 11.8
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

## Install libtorch v2.2.0

```shell
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.2.0+cpu.zip

# .bashrc
export LIBTORCH=/root/libtorch
export CXXFLAGS="-I${LIBTORCH}/include -I${LIBTORCH}/include/torch/csrc/api/include"
export LD_LIBRARY_PATH="${LIBTORCH}/lib:$LD_LIBRARY_PATH"
```