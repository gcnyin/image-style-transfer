from sys import argv
import datetime

import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter import IPAdapter

def main():
    base_model_path = "runwayml/stable-diffusion-v1-5"
    # vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = "models/ip-adapter_sd15.bin"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
    print(f"device: {device}")

    image = Image.open(argv[1])
    g_image = Image.open(argv[2])

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

    images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.3)
    for i in images:
        i.save(f"result-{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}.jpg")

    images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.4)
    for i in images:
        i.save(f"result-{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}.jpg")

    images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.5)
    for i in images:
        i.save(f"result-{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}.jpg")

    images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.6)
    for i in images:
        i.save(f"result-{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}.jpg")

    images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, strength=0.7)
    for i in images:
        i.save(f"result-{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}.jpg")

if __name__ == '__main__':
    main()
