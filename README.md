# Fine-Grained Visual Classification via Internal Ensemble Learning Transformer
Official Pytorch implementation of :

**Article:**  [Fine-Grained Visual Classification via Internal Ensemble Learning Transformer](https://ieeexplore.ieee.org/document/10042971)

**Published in:**  [IEEE Transactions on Multimedia](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6046) ( Early Access ）

If this article is helpful to your work, please cite the following entry:

```latex
@ARTICLE{10042971,
  author={Xu, Qin and Wang, Jiahui and Jiang, Bo and Luo, Bin},
  journal={IEEE Transactions on Multimedia}, 
  title={Fine-Grained Visual Classification Via Internal Ensemble Learning Transformer}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMM.2023.3244340}}
```

or:

Q. Xu, J. Wang, B. Jiang and B. Luo, "Fine-Grained Visual Classification Via Internal Ensemble Learning Transformer," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2023.3244340.

# Abstract

Recently, vision transformers (ViTs) have been investigated in fine-grained visual recognition (FGVC) and are now considered state of the art. However, most ViT-based works ignore the different learning performances of the heads in the multihead self-attention (MHSA) mechanism and its layers. To address these issues, in this paper, we propose a novel internal ensemble learning transformer (IELT) for FGVC. The proposed IELT involves three main modules: multi-head voting (MHV) module, cross-layer refinement (CLR) module, and dynamic selection (DS) module. To solve the problem of the inconsistent performances of multiple heads, we propose the MHV module, which considers all of the heads in each layer as weak learners and votes for tokens of discriminative regions as cross-layer feature based on the attention maps and spatial relationships. To effectively mine the cross-layer feature and suppress the noise, the CLR module is proposed, where the refined feature is extracted and the assist logits operation is developed for the final prediction. In addition, a newly designed DS module adjusts the token selection number at each layer by weighting their contributions of the refined feature. In this way, the idea of ensemble learning is combined with the ViT to improve fine-grained feature representation. The experiments demonstrate that our method achieves competitive results compared with the state of the art on five popular FGVC datasets.

![Network](figures/Network.jpg)!](figures/Network.jpg)

# Experiments Results

| Datasets           | Accuracy (%) | Models | Logs |
| ------------------ | ------------ | ------ | ---- |
| CUB_200_2011       | 91.81        | -      | [link](https://github.com/mobulan/IELT/blob/main/output/logs/CUB.log) |
| Stanford Dogs      | 91.84        | -      | [link](https://github.com/mobulan/IELT/blob/main/output/logs/Dog.log) |
| NABirds            | 90.78        | -      | [link](https://github.com/mobulan/IELT/blob/main/output/logs/NaBirds.log) |
| Oxford 102 Flowers | 99.64        | -      | [link](https://github.com/mobulan/IELT/blob/main/output/logs/Flowers.log) |
| Oxford-IIIT Pet    | 95.29        | -      | [link](https://github.com/mobulan/IELT/blob/main/output/logs/Pet.log) |

# Code Running

## Requirements

python     >= 3.9

pytorch	>= 1.8.1

Apex (optional)

## Training

1. Put the pre-trained ViT model in `pretrained/`, and rename it to `ViT-B_16.npz`, you can download from [ViT pretrained](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz).
2. Select a experiments setting file in `configs/`, and Modify the path of `dataset`.
3. Modify the path in `setup.py` in line 5, and you can change the log name and cuda visible by modify line 13,14.
4. Running the following code according to you pytorch version:

### Single GPU

```bash
python -m main.py
```

### Multiple GPUs

#### If pytorch < 1.12.0

```bash
python -m torch.distributed.launch --nproc_per_node 4 main.py 
```

#### If pytorch >= 1.12.0

```
torchrun --nproc_per_node 4 main.py
```

You need to change the number behind the `-nproc_per_node` to your number of GPUs
