import os
import pprint
import argparse
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image

from configs.demo_config import RunConfig1, RunConfig2, RunConfig3, RunConfig4
from pipe_tome_fpe_hyunjae import tomeV1Pipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
from prompt_utils import PromptParser
import pickle

from diffusers import DDIMScheduler, UNet2DConditionModel, EulerDiscreteScheduler
from torchvision.io import read_image

import warnings

from utils.freeprompt_utils import SelfAttentionControlEdit
from utils.attn_processor import register_attention_control_combined, CombinedAttentionProcessor

warnings.filterwarnings("ignore", category=UserWarning)


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
    config = RunConfig4() #edit this to change the config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stable, prompt_parser = load_model(config, device)
    
    # ------------------parser prompt-------------------------
    if config.use_nlp:
    #     import en_core_web_trf

        # nlp = en_core_web_trf.load()  # load spacy

        # doc = nlp(config.prompt)
        # prompt_parser.set_doc(doc)
        # token_indices = prompt_parser._get_indices(config.prompt)
        # prompt_anchor = prompt_parser._split_prompt(doc)
        # token_indices, prompt_anchor = filter_text(token_indices, prompt_anchor)
        raise NotImplementedError("NLP-based prompt parsing is not implemented yet.")
    else:
        token_indices = config.token_indices
        prompt_anchor = config.prompt_anchor
    # ------------------parser prompt-------------------------

    # token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices

    # FPE
    img_path = config.img_path
    img_dir = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    image_source = load_image(img_path, device)
     
    if os.path.exists(f"{img_dir}/{img_name}_inversion_result.pkl"):
        with open(f"{img_dir}/{img_name}_inversion_result.pkl", "rb") as f:
            start_code, latents_list = pickle.load(f)
            start_code = start_code.to(device)
            latents_list = [l.to(device) for l in latents_list]
    else:
        start_code, latents_list = stable.invert(image_source,
                                                "",
                                                guidance_scale=7.5,
                                                num_inference_steps=50,
                                                return_intermediates=True)
        with open(f"{img_dir}/{img_name}_inversion_result.pkl", "wb") as f:
            pickle.dump((start_code, latents_list), f)
        
    # source_prompt = ""
    target_prompt = [config.prompt] # NOTE(wsgwak): This should be a list of prompts!!!
    # prompts = [source_prompt, target_prompt]
    # start_code = start_code.expand(len(prompts), -1, -1, -1)
    
    NUM_DIFFUSION_STEPS = 50
    self_replace_steps = 0.6

    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        print(f"Original Prompt: {config.prompt}")
        print(f"Anchor Prompt: {prompt_anchor}")
        print(f"Indices of merged tokens: {token_indices}")
        # NOTE(wsgwak): Check it Scheduler should get g. Otherwise the internal copy don't need g.
        g = torch.Generator("cuda").manual_seed(seed) 
        
        tome_controller = AttentionStore() # ToMe controller 
        self_attn_controller = SelfAttentionControlEdit(target_prompt, NUM_DIFFUSION_STEPS, self_replace_steps=self_replace_steps)
        attn_procs, total_hooked_layers = register_attention_control_combined(
            stable,
            tome_controller,
            self_attn_controller,
        )

        image = run_on_prompt(
            image_source=image_source,
            start_code=start_code,
            latents_list=latents_list,
            prompts=target_prompt,
            # prompts=['a cat wearing a shirt and a dog wearing a tie'],
            model=stable,
            tome_attn_store=tome_controller,
            attn_procs=attn_procs,
            token_indices=token_indices,
            prompt_anchor=prompt_anchor,
            seed=g,
            config=config,
        )
        prompt_output_path = config.output_path / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(
            prompt_output_path
            / f'{seed}_{"standard" if config.run_standard_sd else "tome"}.png'
        )
        label = "standard" if config.run_standard_sd else "tome"
        print(f"Image saved to {prompt_output_path / f'{seed}_{label}.png'}")
        images.append(image)

    joined_image = vis_utils.get_image_grid(images)

    joined_image.save(
        config.output_path
        / f'{config.prompt}_{"standard" if config.run_standard_sd else "tome"}.png'
    )
    
    label = "standard" if config.run_standard_sd else "tome"
    filename = f"{config.prompt}_{label}.png"
    print(f"Joined image saved to {config.output_path / filename}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--test", action="store_true", help="Run test")
    args = argparser.parse_args()
    
    main(args)
