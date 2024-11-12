# raggen

<p align="center">
  <img src="https://github.com/user-attachments/assets/1ea28c12-bef7-4811-94fb-8d6448e0b8b0" width=350 />
  Slangin' data and shootin' the RAG.
</p>


## Creating a conda environment
```bash
## Install new conda environment
cd /share/<somewhere>/conda
export CONDA_PKGS_DIRS=$(pwd)/conda_dirs
export CONDA_ENVS_PATH=$(pwd)/conda_envs
conda create --prefix $(pwd)/<env-name> python=3.11.9

## Activate conda environment
conda activate $(pwd)/<env-name>
```

## Installing python environment
```
# Activate conda environment
conda activate /path/to/conda/env

# Install dependencies 
pip3 install -r requirements.txt --no-cache-dir

# Install  cache_replacement package as an interactive python module
pip3 install -e .
```

## Protect your files from HPC garbage collector
```bash
find . -exec touch {} +
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/b8787d88-4004-45f8-8b92-914333a54e79" width=350 />
</p>
