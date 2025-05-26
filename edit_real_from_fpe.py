import os 

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDIMScheduler

# from ReNoise_src.eunms import Model_Type, Scheduler_Type
# from ReNoise_src.utils.enums_utils import get_pipes
# from ReNoise_src.config import RunConfig
from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from ReNoise_main import run as invert

# from Freeprompt.diffuser_utils import FreePromptPipeline
from pipe_tome_fpe import tomePipeline
from Freeprompt.freeprompt_utils import register_attention_control_new
from torchvision.utils import save_image
from torchvision.io import read_image
from Freeprompt.freeprompt import SelfAttentionControlEdit


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model_path = "runwayml/stable-diffusion-v1-5"
model_path = "stabilityai/stable-diffusion-xl-base-1.0"

# control parameters
self_replace_steps = .6
NUM_DIFFUSION_STEPS = 50

# setup
out_dir = "examples/outputs"
SOURCE_IMAGE_PATH = "examples/img/man.jpeg"
source_prompt = ""
target_prompt = 'a silver robot man with yellow umbrella'

# NOTE(wsgwak): check the effect of beta_start, beta_end, beta_schedule
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = tomePipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

# invert the source image
source_image = load_image(SOURCE_IMAGE_PATH, device)
# start_code, latents_list = pipe.invert(source_image,
#                                         source_prompt,
#                                         guidance_scale=7.5,
#                                         num_inference_steps=50,
#                                         return_intermediates=True)
model_type = Model_Type.SDXL
scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)
config = RunConfig(model_type = model_type,
                    num_inference_steps = 50,
                    num_inversion_steps = 50,
                    # num_renoise_steps = 1,
                    scheduler_type = scheduler_type,
                    perform_noise_correction = False,
                    seed = 7865,
                    guidance_scale = 7.5,) # NOTE(wsgwak): Check if 0.0 works better

_, start_code, _, latents_list = invert(source_image,
                                       source_prompt,
                                       config,
                                       pipe_inversion=pipe_inversion,
                                       pipe_inference=pipe_inference,
                                       do_reconstruction=False)


# latents = torch.randn(start_code.shape, device=device)
prompts = [source_prompt, target_prompt]

start_code = start_code.expand(len(prompts), -1, -1, -1)
controller = SelfAttentionControlEdit(prompts, NUM_DIFFUSION_STEPS, self_replace_steps=self_replace_steps)

register_attention_control_new(pipe, controller)

# Note: querying the inversion intermediate features latents_list
# may obtain better reconstruction and editing results
# NOTE(wsgwak): analyze the comment
results = pipe(prompts,
                    latents=start_code,
                    guidance_scale=7.5,
                    ref_intermediate_latents=latents_list)
save_image(results[1], os.path.join(out_dir, str(target_prompt)+'.jpg'))




################## SDXL inversion example ##################
# import torch
# from PIL import Image

# from src.eunms import Model_Type, Scheduler_Type
# from src.utils.enums_utils import get_pipes
# from src.config import RunConfig

# from main import run as invert



# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model_type = Model_Type.SDXL
# scheduler_type = Scheduler_Type.DDIM
# pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

# input_image = Image.open("example_images/lion.jpeg").convert("RGB").resize((1024, 1024))
# prompt = "a lion in the field"

# config = RunConfig(model_type = model_type,
#                     num_inference_steps = 50,
#                     num_inversion_steps = 50,
#                     # num_renoise_steps = 1,
#                     scheduler_type = scheduler_type,
#                     perform_noise_correction = False,
#                     seed = 7865,
#                     guidance_scale = 7.5,)

# _, inv_latent, _, all_latents = invert(input_image,
#                                        prompt,
#                                        config,
#                                        pipe_inversion=pipe_inversion,
#                                        pipe_inference=pipe_inference,
#                                        do_reconstruction=False)

# rec_image = pipe_inference(image = inv_latent,
#                            prompt = prompt,
#                            denoising_start=0.0,
#                            num_inference_steps = config.num_inference_steps,
#                            guidance_scale = 1.0).images[0]

# rec_image.save("lion_reconstructed.jpg")

