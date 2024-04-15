# Training and Testing Texture Similarity Metrics for Structurally Lossless Compression [[Paper]](https://ieeexplore.ieee.org/abstract/document/10438389)

Pytorch version of STSIM metrics. This repository includes code for loading dataset, and the metric training/testing.

## Environment

You can install the environment through `environment.yml`

The code has been tested with python 3.6 and Pytorch 1.7.0. An old Pytorch version is used because the steerable filter is implemented with Pytorch 1.7.0. Some functions are removed in future Pytorch versions.

The metric should support latest Pytorch version if Steerable filter is not used.


## Dataset

The dataset is named TextureGD: a texture geometric distortion database for image compression.

Download link is available here: 
[Google Drive](https://drive.google.com/file/d/1HKp1QdwDi_vWDhrdzKlV4gMXr9KqLJvD/view?usp=sharing)

The dataset includes 22 textures images, their distortions, and annotations from subjects. Only 20 textures are used in the [[Paper]](https://ieeexplore.ieee.org/abstract/document/10438389).

Download the dataset to your computer and change the variable `dataset_dir` in `config/train_STSIM_global.cfg`

## Usage

Training the STSIM metric `python train_global.py --n_textures 22`

Testing the STSIM metric `python test_global.py --n_textures 22`

## Reproduce Results

If you want to reproduce the results in the paper, you can run `./train_global.sh`. The script trains STSIM on the dataset with 5-fold cross validation.

Scripts `./test_global.sh` shows the test results of the metrics with highest performance on evaluation set. The results should be close to the numbers in Table VI in the [[Paper]](https://ieeexplore.ieee.org/abstract/document/10438389).

Remember to change the variable `dataset_dir` in the correspondent configuration files.
