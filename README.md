![Python](https://img.shields.io/badge/python-3.8-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.2.3-green?style=flat-square&logo=tensorflow)

# FlipReID: Closing the Gap Between Training and Inference in Person Re-Identification

## Overview

Since neural networks are data-hungry, incorporating data augmentation in training is a widely adopted technique that enlarges datasets and improves generalization.
On the other hand, aggregating predictions of multiple augmented samples (i.e., test-time augmentation) could boost performance even further.
In the context of person re-identification models, it is common practice to extract embeddings for both the original images and their horizontally flipped variants.
The final representation is the mean of the aforementioned feature vectors.
However, such scheme results in a gap between training and inference, i.e., the mean feature vectors calculated in inference are not part of the training pipeline.
In this study, we devise the FlipReID structure with the flipping loss to address this issue.
More specifically, models using the FlipReID structure are trained on the original images and the flipped images simultaneously, and incorporating the flipping loss minimizes the mean squared error between feature vectors of corresponding image pairs.
Extensive experiments show that our method brings consistent improvements.
In particular, we set a new record for MSMT17 which is the largest person re-identification dataset.
The source code is available at https://github.com/nixingyang/FlipReID.

## Environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda config --set auto_activate_base false
conda create --yes --name TensorFlow2.2 python=3.8
conda activate TensorFlow2.2
conda install --yes cudatoolkit=10.1 cudnn=7.6 -c nvidia
conda install --yes cython matplotlib numpy=1.18 pandas pydot scikit-learn
pip install tensorflow==2.2.3
pip install tf2cv
pip install opencv-python
pip install albumentations --no-binary imgaug,albumentations
```

## Training

```bash
python3 -u solution.py --dataset_name "Market1501" --use_horizontal_flipping_inside_model --nouse_horizontal_flipping_in_evaluation --steps_per_epoch 200 --epoch_num 200
```

- To train on other datasets, replace `"Market1501"` with `"DukeMTMC_reID"` or `"MSMT17"`.
- Specify the arguments for the backbone model:
```bash
--backbone_model_name "resnet50" --learning_rate_start 2e-4 --learning_rate_end 2e-4 --learning_rate_base 2e-4 --learning_rate_lower_bound 2e-6 --kernel_regularization_factor 0.0005 --bias_regularization_factor 0.0005 --gamma_regularization_factor 0.0005 --beta_regularization_factor 0.0005
```
```bash
--backbone_model_name "ibn_resnet50" --learning_rate_start 3e-4 --learning_rate_end 3e-4 --learning_rate_base 3e-4 --learning_rate_lower_bound 3e-6 --kernel_regularization_factor 0.0005 --bias_regularization_factor 0.0005 --gamma_regularization_factor 0.0005 --beta_regularization_factor 0.0005
```
```bash
--backbone_model_name "resnesta50" --learning_rate_start 3e-4 --learning_rate_end 3e-4 --learning_rate_base 3e-4 --learning_rate_lower_bound 3e-6 --kernel_regularization_factor 0.0010 --bias_regularization_factor 0.0010 --gamma_regularization_factor 0.0010 --beta_regularization_factor 0.0010
```
- To evaluate on a subset of the complete test set, append `--testing_size 0.5` to the command. Alternatively, you may turn this feature off by using `--testing_size 0.0`.

## Evaluation

```bash
python3 -u solution.py --dataset_name "Market1501" --backbone_model_name "resnesta50" --pretrained_model_file_path "?.h5" --use_horizontal_flipping_inside_model --use_horizontal_flipping_in_evaluation --output_folder_path "evaluation_only" --evaluation_only --freeze_backbone_for_N_epochs 0 --testing_size 1.0 --evaluate_testing_every_N_epochs 1
```

- Fill in the `pretrained_model_file_path` argument using the h5 file obtained during training.
- To use the re-ranking method, append `--use_re_ranking` to the command.

## Model Zoo

| Dataset | Backbone | mAP | Weights |
| - | - | - |- |
| Market1501 | ResNeSt50 | 89.6 | [Link](https://1drv.ms/u/s!Av-teFsyVR6Wjl8ukmZLB4D1BoGE) |
| DukeMTMC_reID | ResNeSt50 | 81.5 | [Link](https://1drv.ms/u/s!Av-teFsyVR6WjmCrhd3tjKd-FcSP) |
| MSMT17 | ResNeSt50 | 68.0 | [Link](https://1drv.ms/u/s!Av-teFsyVR6Wjl6WbcmC3cK7KBQy) |

## Acknowledgements

- Evaluation Metrics are adapted from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid/blob/v1.0.6/torchreid/metrics/rank_cylib/rank_cy.pyx).
- Re-Ranking is adapted from [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking/blob/master/python-version/re_ranking_ranklist.py).
- Random Grayscale Patch Replacement is adapted from [Data-Augmentation](https://github.com/finger-monkey/Data-Augmentation/blob/main/trans_gray.py).
- Random Erasing is adapted from [Random-Erasing](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py).
- Triplet Loss is adapted from [triplet-reid](https://github.com/VisualComputingInstitute/triplet-reid/blob/master/loss.py).

## Citation

Please consider citing [this work](https://ieeexplore.ieee.org/document/9484010) if it helps your research.

```
@inproceedings{ni2021flipreid,
  author={Ni, Xingyang and Rahtu, Esa},
  booktitle={2021 9th European Workshop on Visual Information Processing (EUVIP)},
  title={FlipReID: Closing the Gap Between Training and Inference in Person Re-Identification},
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/EUVIP50544.2021.9484010}
}
```
