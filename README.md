This repository is our paper "Training and Testing Texture Similarity Metrics
for Image Compression" (submitted).

# Dataset 

The dataset is name TexTexGD: a geometric distortion database for image compression.

Download link is available [here](https://drive.google.com/file/d/1R0jN3YAN0B-2ou6cFuHGr_ZPwiqIOWNv/view?usp=sharing).

Download it to your computer and change the variable `dataset_dir` in `config/test_STSIM_global.cfg`

# Usage
Clone this repo and run
`python test_global.py`,
you should expect following results

`STSIM-M (trained) test: {'PLCC': 0.939, 'SRCC': 0.929, 'KRCC': 0.798}` 

which is the same results in table 6 in the paper.

There are a few other pre-trained weights under folder `weights`, you can modify `config/test_STSIM_global.cfg`
and run `python test_global.py` to test different metric weights.
