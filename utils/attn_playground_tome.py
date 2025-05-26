import os

import torch
import torch.nn.functional as F
import numpy as np
import diffusers

# from attn_processor import AttentionStore, AttendExciteCrossAttnProcessor, SelfAttentionControlEdit
# from freeprompt_utils import register_attention_control_new, SelfAttentionControlEdit
from utils.ptp_utils import register_attention_control, AttentionStore

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
controller = AttentionStore()
register_attention_control(
    model,
    controller,
)

# ----------------------------------------