## MMTM: Multimodal Transfer Module for CNN Fusion

Code for the paper [MMTM: Multimodal Transfer Module for CNN Fusion](https://arxiv.org/abs/1911.08670). This is a reimplementation of the original MMTM code to reproduce the results on NTU RGB+D dataset in Table 5 of the paper.

If you use this code, please cite the paper:

```
@inproceedings{vaezi20mmtm,
 author = {Vaezi Joze, Hamid Reza and Shaban, Amirreza and Iuzzolino, Michael L. and Koishida, Kazuhito},
 booktitle = {Conference on Computer Vision and Pattern Recognition ({CVPR})},
 title = {MMTM: Multimodal Transfer Module for CNN Fusion},
 year = {2020}
}
```

## Installation
This code has been tested on Ubuntu 16.04 with Python 3.8.3 and PyTorch 1.4.0.
* Install [Pytorch 1.4.0](https://pytorch.org).
* Install [tqdm](https://github.com/tqdm/tqdm) by running `pip install tqdm`.
* Install opencv by running 'pip install opencv-python`.
* Install matplotlib by running `pip install matplotlib`.
* Install sklearn by running `pip install sklearn`.

## Download the pre-trained checkpoints and prepare NTU RGB+D dataset
* Download and uncompress the [checkpoints](https://gtvault-my.sharepoint.com/:u:/g/personal/ashaban6_gatech_edu/EZQR-QfpPqZPnK_ClGGkbtYBuDqWgWUdlsdun5p316uHIQ?e=1Nz8FI) and place them in 'ROOT/checkpoint' dicrectory.
* Download [NTU RGB+D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset.
* Copy all skeleton files to `ROOT/NUT/nturgbd_skeletons/` directory. 
* Change all video clips resolution to 256x256 30fps and copy them to `ROOT/NTU/nturgbd_rgb/avi_256x256_30/` directory.

## Evaluation
* Run `python main_mmtm_ntu.py --datadir ROOT/NTU --checkpointdir ROOT/checkpoints --test_cp fusion_mmtm_epoch_8_val_loss_0.1873.checkpoint --no_bad_skel`.
* You can reduce the batch size if run out of memeory e.g. `--batchsize 1`.
* Add '--use_dataparallel' to use multiple GPUs.

## Training
* Run `python main_mmtm_ntu.py --datadir ROOT/NTU --checkpointdir ROOT/checkpoints --train --ske_cp skeleton_32frames_85.24.checkpoint --rgb_cp rgb_8frames_83.91.checkpoint`.
* We have trained the model with `--batchsize 20` and `--use_dataparallel` options on 4 GPUs.
