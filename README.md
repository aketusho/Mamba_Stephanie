# HSMF_PointMamba

## 1. Requirements
PyTorch >= 1.7.0 < 1.11.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
argparse
easydict
h5py
matplotlib
numpy
open3d==0.9
opencv-python
pyyaml
scipy
tensorboardX
timm==0.4.5
tqdm
transforms3d
termcolor
```

Install Multiscale Mamba
```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Install Surface Representation 

```
# Using original file directly (need the same folder structure)
# RepSurf for Classification
https://github.com/hancyran/RepSurf/tree/main/classification

# Installation modules
cd modules/pointops
python3 setup.py install
cd -
```

## 2. Datasets

See ScanObjectNN and ModelNet40 in [DATASET.md](./DATASET.md) for details.
