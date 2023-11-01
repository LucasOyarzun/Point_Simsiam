# Point Simsiam

Code for Self-supervised Learning for 3D Representations

## Requirements

### Installation
Create a conda environment and install basic dependencies:
```bash
git clone https://github.com/LucasOyarzun/Point_Simsiam.git
cd Point_Simsiam

conda create -n point-simsiam python=3.8
conda activate point-simsiam

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
# e.g., conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3

pip install -r requirements.txt
```
Install GPU-related packages:
```bash
# Chamfer Distance and EMD
cd ./extensions/chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
### Datasets
For pre-training and fine-tuning, please follow [DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) to install ShapeNet, ModelNet40, ScanObjectNN, and ShapeNetPart datasets, referring to Point-BERT. For Linear SVM evaluation, download the official [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) dataset and put the unzip folder under `data/`.

The final directory structure should be:
```
│Point_Simsiam/
├──cfgs/
├──datasets/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──modelnet40_ply_hdf5_2048/  # Specially for Linear SVM
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──...
```


## Get Started

### Pre-training
Point-M2AE is pre-trained on ShapeNet dataset with the config file `cfgs/pre-training/point-m2ae.yaml`. Run:
```bash
python main.py --config cfgs/PointSimsiam/pretrain.yaml --exp_name simsiam_pretrain
```

To evaluate the pre-trained Point Simsiam by **Linear SVM**, Run:

For ModelNet40, run:
```bash
python main.py --config cfgs/PointSimsiam/linear_probing_modelnet40.yaml --linear_probing modelnet40 --linear_model <knn|svm> --exp_name simsiam_test_svm --ckpts experiments/.../ckpt-last.pth
```

### Fine-tuning
Please create a folder `ckpts/` and download the [pre-train.pth](https://drive.google.com/file/d/1HyUEv04V2K6vMaR0P7WksuoiMtoXx1fM/view?usp=share_link) into it. The fine-tuning configs are in `cfgs/fine-tuning/`.

For ModelNet40, run:
```bash
python main.py --config cfgs/PointSimsiam/finetune_modelnet.yaml --finetune_model --exp_name simsiam_finetune --ckpts experiments/.../ckpt-last.pth
```

For the three splits of ScanObjectNN, run:

```bash
python main.py --config cfgs/PointSimsiam/scan_pb.yaml --finetune_model --exp_name simsiam_scan_pb --ckpts experiments/.../ckpt-last.pth
```
```bash
python main.py --config cfgs/PointSimsiam/scan_obj.yaml --finetune_model --exp_name simsiam_scan_obj --ckpts experiments/.../ckpt-last.pth
```
```bash
python main.py --config cfgs/PointSimsiam/scan_obj-bg.yaml --finetune_model --exp_name simsiam_scan_obj-bg --ckpts experiments/.../ckpt-last.pth
```

For ShapeNetPart, first into the `segmentation/` folder, and run:
```bash
cd segmentation
python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300
```

## Acknowledgement
This repo benefits from:

* [Point-BERT](https://github.com/lulutang0608/Point-BERT)
* [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)
* [Point-M2AE(https://github.com/ZrrSkywalker/Point-M2AE)]
* [PointNet_Pointnet2](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
* [DGCNN](https://github.com/WangYueFt/dgcnn)
* [PointMLP](https://github.com/ma-xu/pointMLP-pytorch)

Thanks for their wonderful works.

## Citation

Coming soon...

## Contact
If you have any question about this project, please feel free to contact lucas.oyarzun@ing.uchile.cl