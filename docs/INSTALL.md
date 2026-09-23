# Installation

Our codebase is built upon **Python 3.12, PyTorch 2.4.0 (recommended)**.

## Setup

```shell
conda create -n SimMOT python=3.12       # suggest to use virtual envs
conda activate SimMOT
# PyTorch:
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# Other dependencies:
conda install pyyaml tqdm matplotlib scipy pandas
pip install wandb accelerate einops
```

## Compile Deformable Attention

```shell
cd models/ops/
sh make.sh
# [Optional] After compiled, you can use following script to test it:
python test.py
```
