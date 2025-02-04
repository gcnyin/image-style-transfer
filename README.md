# Image style transfer tool

This script is based on StableDiffusionImg2ImgPipeline and IPAdapter, used for generating new images through image-to-image generation, combining the functions of IP-Adapter. Users can specify input and guidance images, adjust generation strength and other parameters to generate output images with different styles or features.

## Dependency environment

- Python 3.12
- PyTorch (Recommended to use a GPU environment with CUDA support, you can also use a CPU or MPS environment for M1 chips)
- ip_adapter library (to be installed via `install.sh`)

How to use

Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./install.sh
```

Run the script

```shell
python3 style_transfer.py examples/style-002.jpg examples/content-001.jpg
```

Parameter description

Input image: The base image for generation.
Guiding image: Used to guide the direction or style of generation.

The generated image will be saved in the directory where the script is running, with the file name as result-YYYYMMDDTHHMMSSZ.jpg, where the timestamp is in UTC time.

# 图像风格迁移工具

本脚本基于 StableDiffusionImg2ImgPipeline 和 IPAdapter，用于通过图像到图像的生成方式，结合 IP-Adapter 的功能，生成新的图像。用户可以通过指定输入图像和引导图像，调整生成强度等参数，生成具有不同风格或特征的输出图像。

## 依赖环境

- Python 3.12
- PyTorch（建议使用 CUDA 支持的 GPU 环境，也可使用 CPU 或 M1 芯片的 MPS 环境）
- ip_adapter 库（需通过`install.sh`安装）

## 使用方法

### 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./install.sh
```

### 运行脚本

```shell
python3 style_transfer.py examples/style-002.jpg examples/content-001.jpg
```

参数说明

- 输入图像：作为生成的基础图像。
- 引导图像：用于引导生成的方向或风格。

生成的图像将保存在脚本运行目录下，文件名为 result-YYYYMMDDTHHMMSSZ.jpg，其中时间戳为 UTC 时间。
