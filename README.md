<p align="center">
<h1 align="center">GAP: Gaussianize Any Point Clouds with Text Guidance <br>
(ICCV 2025)</h1>
<p align="center">
    <a href="https://weiqi-zhang.github.io/"><strong>Weiqi Zhang*</strong></a>
    ·
    <a href="https://junshengzhou.github.io/"><strong>Junsheng Zhou*†</strong></a>
    ·
    <a href="https://github.com/mts246/"><strong>Haotian Geng*</strong></a>
    ·
    <a href="https://wen-yuan-zhang.github.io/"><strong>Wenyuan Zhang</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu†</strong></a>
</p>
<p align="center"><strong>(* Equal Contribution † Corresponding Author)</strong></p>
<h3 align="center"><a href="https://yushen-liu.github.io/main/pdf/LiuYS_ICCV25_GAP.pdf">Paper</a> | <a href="https://weiqi-zhang.github.io/GAP/">Project Page</a></h3>
<div align="center"></div>
</p>
<p align="center">
    <img src="figs/teaser.png" width="780" />
</p>



We will release the code of the paper <a href="https://weiqi-zhang.github.io/GAP/">GAP: Gaussianize Any Point Clouds with Text Guidance</a> in this repository.

## Abstract

<p>
In this work, we introduce GAP, a novel method for transforming raw, colorless 3D point clouds into high-quality Gaussian representations guided by text. Although 3D Gaussian Splatting (3DGS) has demonstrated impressive rendering capabilities, it remains challenging to convert sparse and uncolored point clouds into meaningful Gaussian forms. GAP effectively addresses this issue, performing robustly across a wide range of scenarios, including synthetic datasets, real-world scans, and large-scale scenes. It provides a reliable and versatile solution for point-to-Gaussian conversion.
          </p>


## Method

<p align="center">
  <img src="figs/method.png" width="780" />
</p>
<p style="margin-top: 30px">
<strong>Overview of GAP.</strong>
            <strong>(a)</strong> We rasterize the Gaussians through an unprocessed view, where a depth-aware image diffusion model is used to generate consistent appearances using the rendered depth and mask with text guidance. The mask is dynamically classified as generate, keep, or update based on viewing conditions. <strong>(b)</strong> The Gaussian optimization includes three constraints: the Distance Loss and Scale Loss introduced to ensure geometric accuracy, and the Rendering Loss that ensures high-quality appearance. <strong>(c)</strong> The Gaussian inpainting strategy which diffuses the geometric and appearance information from visible regions to hard-to-observe areas, considering local density, spatial proximity and normal consistency.





## Generation Results

### Visual Comparison of Text-Guided Generation

<img src="./figs/text.png" class="center">

### Point to Gaussian

<img src="./figs/point.png" class="center">

### Results on Corrupted Data

<img src="./figs/corrupted.png" class="center">

### Scale to Secene Level

<img src="./figs/scene.png" class="center">

## Visualization Results

<img src="./figs/text.gif" class="center">



## Citation

If you find our code or paper useful, please consider citing

    @inproceedings{gap,
          title={GAP: Gaussianize Any Point Clouds with Text Guidance},
          author={Zhang, Weiqi and Zhou, Junsheng and Geng, Haotian and Zhang, Wenyuan and Liu, Yu-Shen},
          booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
          year={2025}
        }
