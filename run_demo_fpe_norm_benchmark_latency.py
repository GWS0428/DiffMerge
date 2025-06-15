import os
import json
import pprint
import warnings
import argparse
import time
from typing import List
from pathlib import Path 
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from configs.demo_config import RunConfig1, RunConfig2, RunConfig3, RunConfig4
from pipe_tome_fpe_hyunjae import tomeV1Pipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
from prompt_utils import PromptParser
import pickle

from diffusers import DDIMScheduler, UNet2DConditionModel, EulerDiscreteScheduler
from torchvision.io import read_image


# from Freeprompt.freeprompt import SelfAttentionControlEdit
from utils.freeprompt_utils import SelfAttentionControlEdit
# from Freeprompt.freeprompt_utils import register_attention_control_new
from utils.attn_processor import register_attention_control_combined, CombinedAttentionProcessor
from metrics import calculate_clip_score, calculate_directional_similarity


warnings.filterwarnings("ignore", category=UserWarning)


def dict_to_namespace(data):
    """Recursively converts a dictionary to a SimpleNamespace."""
    if isinstance(data, dict):
        # Handle special cases if needed, e.g., convert 'output_path' string to Path
        # For example:
        # if 'output_path' in data and isinstance(data['output_path'], str):
        #     data['output_path'] = Path(data['output_path'])
        if not all(isinstance(k, str) for k in data.keys()):
            return {k: dict_to_namespace(v) for k, v in data.items()}
        
        return SimpleNamespace(**{
            key: dict_to_namespace(value)
            for key, value in data.items()
        })
    elif isinstance(data, list):
         return [dict_to_namespace(item) for item in data]
    else:
        return data


def load_config_json(filepath):
    """Loads a configuration from a JSON file into a SimpleNamespace."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    # Handle Path conversion if necessary after loading
    if 'output_path' in config_dict and isinstance(config_dict['output_path'], str):
         config_dict['output_path'] = Path(config_dict['output_path'])
         
    if 'thresholds' in config_dict:
        config_dict['thresholds'] = {int(k): v for k, v in config_dict['thresholds'].items()}


    return dict_to_namespace(config_dict)


def read_prompt(path):
    with open(path, "r") as f:
        prompt_ls = f.readlines()

    all_prompt = []

    for idx, prompt in enumerate(prompt_ls):
        prompt = prompt.replace("\n", "")
        all_prompt.append([idx, prompt])
    return all_prompt


def load_model(config, device):

    # stable_diffusion_version = "stabilityai/stable-diffusion-xl-base-1.0"
    stable_diffusion_version = "runwayml/stable-diffusion-v1-5"

    if hasattr(config, "model_path") and config.model_path is not None:
        stable_diffusion_version = config.model_path
    
    # FPE
    # NOTE(wsgwak): check the effect of scheduler. Current code adopt FPE setup.
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # from transformers import CLIPTokenizer
    # tokenizer_1 = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-g-14")

    stable = tomeV1Pipeline.from_pretrained(
        stable_diffusion_version,
        # torch_dtype=torch.float16,
        # variant="fp16",
        safety_checker=None,
        scheduler=scheduler, 
        addition_embed_type=None,
    ).to(device)
    stable.unet.config.addition_embed_type=None
    print(stable.unet.config.addition_embed_type)
    # exit()
    
    # TEST(wsgwak)
    # stable.scheduler = EulerDiscreteScheduler.from_config(stable.scheduler.config)
    
    # stable.enable_xformers_memory_efficient_attention()
    stable.unet.requires_grad_(False)
    stable.vae.requires_grad_(False)
    # stable.enable_model_cpu_offload()
    
    # Inject tokenizers manually
    # stable.tokenizer = tokenizer_1
    # stable.tokenizer_2 = tokenizer_2

    prompt_parser = PromptParser(stable_diffusion_version)

    return stable, prompt_parser


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {
        idx: stable.tokenizer.decode(t)
        for idx, t in enumerate(stable.tokenizer(prompt)["input_ids"])
        if 0 < idx < len(stable.tokenizer(prompt)["input_ids"]) - 1
    }
    pprint.pprint(token_idx_to_word)
    token_indices = input(
        "Please enter the a comma-separated list indices of the tokens you wish to "
        "alter (e.g., 2,5): "
    )
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(
    image_source: torch.Tensor,
    start_code: torch.Tensor,
    latents_list: List[torch.Tensor],
    prompts: List[str],
    model: tomeV1Pipeline,
    tome_attn_store: AttentionStore,
    attn_procs: List[CombinedAttentionProcessor],
    token_indices: List[int],
    prompt_anchor: List[str],
    seed: torch.Generator,
    config,
) -> Image.Image:
    
    outputs = model(
        latents=start_code, # added
        ref_intermediate_latents=latents_list, # added
        prompt=prompts,
        guidance_scale=config.guidance_scale,
        generator=seed,
        num_inference_steps=config.n_inference_steps,
        attention_store=tome_attn_store,
        attn_procs=attn_procs,
        indices_to_alter=token_indices,
        prompt_anchor=prompt_anchor,
        attention_res=config.attention_res,
        run_standard_sd=config.run_standard_sd,
        thresholds=config.thresholds,
        scale_factor=config.scale_factor,
        scale_range=config.scale_range,
        prompt3=config.prompt_merged,
        prompt_length=config.prompt_length,
        token_refinement_steps=config.token_refinement_steps,
        attention_refinement_steps=config.attention_refinement_steps,
        tome_control_steps=config.tome_control_steps,
        eot_replace_step=config.eot_replace_step,
        use_pose_loss=config.use_pose_loss,
        # negative_prompt="low res, ugly, blurry, artifact, unreal",
        negative_prompt=[""],
        use_fpe=config.use_fpe,
    )
    if config.use_fpe:
        image = outputs.images[1]
    else:
        image = outputs.images[0]
    return image


def filter_text(token_indices, prompt_anchor):
    final_idx = []
    final_prompt = []
    for i, idx in enumerate(token_indices):
        if len(idx[1]) == 0:
            continue
        final_idx.append(idx)
        final_prompt.append(prompt_anchor[i])
    return final_idx, final_prompt




def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


def main(args):
    config = load_config_json(args.config_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stable, prompt_parser = load_model(config, device) if not args.metric_only else (None, None)

    if config.use_nlp:
        raise NotImplementedError("NLP-based prompt parsing is not implemented yet.")
    else:
        token_indices = config.token_indices
        prompt_anchor = config.prompt_anchor

    # FPE inversion
    img_path = config.img_path
    img_dir = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    image_source = load_image(img_path, device)

    # get inverted latents
    for _ in range(3):
        start_code, latents_list = stable.invert(image_source,
                                                    "",
                                                    guidance_scale=7.5,
                                                    num_inference_steps=50,
                                                    return_intermediates=True)
    time_inversion = 0
    for _ in range(10):
        time_inversion_start = time.time()
        start_code, latents_list = stable.invert(image_source,
                                                    "",
                                                    guidance_scale=7.5,
                                                    num_inference_steps=50,
                                                    return_intermediates=True)
        time_inversion_end = time.time()
        time_inversion += (time_inversion_end - time_inversion_start)
    time_inversion_avg = time_inversion / 10
    print(f"Average inversion time: {time_inversion_avg:.4f} seconds")
    
    target_prompt = [config.prompt]
    NUM_DIFFUSION_STEPS = 50
    self_replace_steps = config.self_replace_steps

    if len(config.seeds) != 1:
        raise ValueError("Only one seed is supported for latency testing.")

    seed = config.seeds[0]
    g = torch.Generator("cuda").manual_seed(seed)

    tome_controller = AttentionStore()
    self_attn_controller = SelfAttentionControlEdit(target_prompt, NUM_DIFFUSION_STEPS, self_replace_steps=self_replace_steps)
    attn_procs, total_hooked_layers = register_attention_control_combined(
        stable,
        tome_controller,
        self_attn_controller,
    )

    print("\nRunning warm-up steps...")
    for i in range(3):
        _ = run_on_prompt(
            image_source=image_source,
            start_code=start_code,
            latents_list=latents_list,
            prompts=target_prompt,
            model=stable,
            tome_attn_store=tome_controller,
            attn_procs=attn_procs,
            token_indices=token_indices,
            prompt_anchor=prompt_anchor,
            seed=g,
            config=config,
        )
        print(f"Warm-up run {i + 1}/3 completed.")

    print("\nRunning timed inference...")
    durations = []
    for i in range(10):
        start_time = time.time()
        _ = run_on_prompt(
            image_source=image_source,
            start_code=start_code,
            latents_list=latents_list,
            prompts=target_prompt,
            model=stable,
            tome_attn_store=tome_controller,
            attn_procs=attn_procs,
            token_indices=token_indices,
            prompt_anchor=prompt_anchor,
            seed=g,
            config=config,
        )
        end_time = time.time()
        duration = end_time - start_time
        durations.append(duration)
        print(f"Run {i + 1}/10 latency: {duration:.4f} seconds")

    avg_latency = sum(durations) / len(durations)
    avg_total = avg_latency + time_inversion_avg  # Include inversion time in average
    print(f"\n--- Latency Statistics ---")
    print(f"Total Runs: {len(durations)}")
    print(f"Average Latency per Image Inversion: {time_inversion_avg:.4f} seconds")
    print(f"Average Latency per Image Edit: {avg_latency:.4f} seconds")
    print(f"Min Latency: {min(durations):.4f} seconds")
    print(f"Max Latency: {max(durations):.4f} seconds")
    print(f"Average Total Latency (including inversion): {avg_total:.4f} seconds")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_file", type=str, default="configs/config_default.json",)
    argparser.add_argument("--metric", action="store_true", help="Calculate metrics like CLIP Score and CDS.")
    argparser.add_argument("--metric_only", action="store_true", help="Only calculate metrics without running the model.")
    argparser.add_argument("--rerun", action="store_true", help="Rerun the model even if images already exist.")
    args = argparser.parse_args()
    
    main(args)
