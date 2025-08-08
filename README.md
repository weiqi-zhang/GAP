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


## Setup

1. **Create a conda environment with Python 3.9:**

```bash
conda create -n gap python=3.9
conda activate gap
```

2. **Install PyTorch with CUDA support:**

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

3. **Install other Python dependencies:**

```bash
pip install -r requirements.txt
```

4. **Model Download**

To use the ControlNet Depth2img model, please download `control_sd15_depth.pth` from the [hugging face page](https://huggingface.co/lllyasviel/ControlNet/tree/main/models), and put it under `models/ControlNet/models/`. And, please download `dpt_hybrid-midas-501f0c75.pt` from the [hugging face page](https://huggingface.co/lllyasviel/Annotators/blob/main/dpt_hybrid-midas-501f0c75.pt), and put it under `models/ControlNet/annotator/ckpts`.



## Usage

### Data Preparation

GAP supports point cloud files in the following formats:
- `.ply` files with xyz coordinates
- `.xyz` files with plain text coordinates
- `.npy` files with numpy arrays

Your input directory should be organized as:
```
data/
├── your_object/
│   └── your_object.ply  # Point cloud file
└── another_object/
    └── another_object.ply
```

### Basic Usage

#### Quick Start

Run one of the provided example scripts:

```bash
bash bash/backpack.sh
```

#### Custom Generation

For your own point cloud data, run the main script with custom parameters:

```bash
python scripts/generate_gaussian_text.py \
    --input_dir data/your_object \
    --output_dir outputs/your_object \
    --pc_name your_object \
    --pc_file your_object.ply \
    --prompt "A detailed description of your object" \
    --ddim_steps 50 \
    --num_viewpoints 40 \
    --viewpoint_mode predefined \
    --update_steps 30 \
    --seed 42
```

#### Key Parameters

- `--input_dir`: Directory containing your point cloud file
- `--output_dir`: Where to save the generated results
- `--pc_name`: Name identifier for your object
- `--pc_file`: Point cloud filename (supports .ply, .xyz, .npy)
- `--prompt`: Text description to guide the generation process
- `--ddim_steps`: Number of diffusion steps (20-50, higher = better quality)
- `--num_viewpoints`: Number of viewpoints for generation (8-40)
- `--viewpoint_mode`: `predefined` or `hemisphere` 
- `--new_strength`: Strength for generating new regions (0.0-1.0)
- `--update_strength`: Strength for updating existing regions (0.0-1.0) 
- `--device`: GPU type (`a6000` or `2080` for memory optimization)
- `--seed`: Random seed for reproducible results

### Output

After successful generation, the main result is `final.ply` which contains the colored Gaussian representation.

### Note

- Ensure your point cloud follows the standard orientation: **Y-axis up**, facing **+Z direction**.
- For optimal results, you may adjust the predefined viewpoints based on your specific object geometry.



## Acknowledgements

This project is built upon [2DGS](https://github.com/hbb1/2d-gaussian-splatting), [CAP-UDF](https://github.com/junshengzhou/CAP-UDF) and [Text2Tex](https://github.com/daveredrum/Text2Tex). We thank all the authors for their great repos.



## Citation

If you find our code or paper useful, please consider citing

    @inproceedings{gap,
          title={GAP: Gaussianize Any Point Clouds with Text Guidance},
          author={Zhang, Weiqi and Zhou, Junsheng and Geng, Haotian and Zhang, Wenyuan and Liu, Yu-Shen},
          booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
          year={2025}
        }
