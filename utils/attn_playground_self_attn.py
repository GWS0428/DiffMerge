import os

import torch
import torch.nn.functional as F
import numpy as np
import diffusers

# from attn_processor import AttentionStore, AttendExciteCrossAttnProcessor, SelfAttentionControlEdit
from freeprompt_utils import register_attention_control_new, SelfAttentionControlEdit

from run_demo_fpe import get_dummy_inversion_output
from src.eunms import Model_Type, Scheduler_Type

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


model = diffusers.StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    # "runwayml/stable-diffusion-v1-5",
    # torch_dtype=torch.float16,
    safety_checker=None,
).to(device)

# ----------------- TOME -----------------
prompts=["", "a dog wearing hat"]
controller = SelfAttentionControlEdit(
    prompts=prompts,
    num_steps=50,
    self_replace_steps=.6,
)

register_attention_control_new(
    model,
    controller,
)
# _, start_code, _, latents_list = get_dummy_inversion_output(
#     model_type=Model_Type.SD15,
#     num_inversion_steps=50,
#     dtype=torch.float32,
# )
# start_code = start_code.expand(2, -1, -1, -1)
# model(prompts,
#                     latents=start_code,
#                     guidance_scale=7.5,
#                     ref_intermediate_latents=latents_list)
# ---------------------------------------------

# total_attn_count = 0
# for name, processor in model.unet.attn_processors.items():
#     # print(name)
#     # print(processor)
#     # print(processor.__class__.__name__)
#     # # print(processor.res)
#     # print("=====================================")
#     total_attn_count += 1
# print(f"Total attention count: {total_attn_count}")

# def register_recr(net_, count, place_in_unet):
#     # print(net_.__class__.__name__)
#     if net_.__class__.__name__ == 'Attention':
#         net_.forward = ca_forward(net_, place_in_unet)
#         return count + 1
#     elif hasattr(net_, 'children'):
#         for net__ in net_.children():
#             count = register_recr(net__, count, place_in_unet)
#     return count
    
# sub_nets = model.unet.named_children()
# for net in sub_nets:
#     if "down" in net[0]:
#         cross_att_count += register_recr(net[1], 0, "down")
#     elif "up" in net[0]:
#         cross_att_count += register_recr(net[1], 0, "up")
#     elif "mid" in net[0]:
#         cross_att_count += register_recr(net[1], 0, "mid")