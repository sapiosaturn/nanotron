#!/bin/bash
# Note: generally this is used on runpod instances, other ones might have different ways of using apt (like sudo might be needed)
echo "Updating apt"
apt update
echo "Installing dependencies"
apt install wget curl vim nvtop -y
echo "Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
# sync uv
echo "Syncing uv"
uv sync
# download and install precompiled flash-attn
echo "Downloading flash-attn"
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# flash linear attention stuff
echo "Installing flash-linear-attention"
uv pip install -U git+https://github.com/fla-org/flash-linear-attention
uv pip install "numpy<2" #to not break nanotron
uv pip install "causal-conv1d>=1.4.0" --no-build-isolation
# activate venv
echo "Activating venv"
source .venv/bin/activate
echo "Done"
