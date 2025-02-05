import argparse
import datetime

import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter import IPAdapter

def main():
    parser = argparse.ArgumentParser(description="Image style transfer tool 图像风格迁移工具")
    parser.add_argument("-c", "--content_file", help="Input image: The base image for generation. 输入图像：作为生成的基础图像", required=True)
    parser.add_argument("-s","--style_file", help="Guiding image: Used to guide the direction or style of generation. 引导图像：用于引导生成的方向或风格", required=True)
    args = parser.parse_args()

    base_model_path = "runwayml/stable-diffusion-v1-5"
    # vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = "models/ip-adapter_sd15.bin"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

    image = Image.open(args.style_file)
    g_image = Image.open(args.content_file)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    # vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # load SD pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        # vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    # load ip-adapter
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    # images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.3)
    # for i in images:
    #     i.save(f"result-{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}.jpg")

    # images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.4)
    # for i in images:
    #     i.save(f"result-{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}.jpg")

    images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.5)
    for i in images:
        i.save(f"result-{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}-05.jpg")

    # images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.6)
    # for i in images:
    #     i.save(f"result-{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}.jpg")

    # images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.7)
    # for i in images:
    #     i.save(f"result-{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}.jpg")

if __name__ == '__main__':
    main()
