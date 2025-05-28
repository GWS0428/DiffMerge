import sys

import torch
import torch.nn as nn
from diffusers.models.attention import Attention
from typing import List, Dict, Any

# Assume AttentionStore, AttendExciteCrossAttnProcessor, SelfAttentionControlEdit
# and potentially AttentionControl base class are available for import from your project's utils file(s)
# e.g., from utils.ptp_utils import AttentionStore, AttendExciteCrossAttnProcessor, SelfAttentionControlEdit


# Define the CombinedAttentionProcessor
class CombinedAttentionProcessor(nn.Module):
    def __init__(self, tome_controller, self_attn_controller, place_in_unet, tome_control_point):
        super().__init__()
        self.adopt_self_attn = False  # This will be dynamically updated
        self.tome_controller = tome_controller
        self.self_attn_controller = self_attn_controller
        self.place_in_unet = place_in_unet
        self.tome_control_point = tome_control_point
        
    def set_custom_param(self, value):
        self.adopt_self_attn = value

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor, encoder_hidden_states=None,
                    attention_mask = None,temb = None,scale: float = 1.0,) -> torch.Tensor:
        residual = hidden_states
        is_cross = encoder_hidden_states is not None

        args = (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states,)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states,)
        value = attn.to_v(encoder_hidden_states,)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # FIXME(wsgwak): Add a logic to avoid graident error at this point
        if self.adopt_self_attn:
            attention_probs = self.self_attn_controller(attention_probs, is_cross, self.place_in_unet)
        if self.tome_control_point:
            self.tome_controller(attention_probs, is_cross, self.place_in_unet) 
        # self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states,)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# Define the registration function
def register_attention_control_combined(model, tome_controller, self_attn_controller):
    tome_attn_greenlist = ["up_blocks.0.attentions.1.transformer_blocks.1.attn2.processor",
                           "up_blocks.0.attentions.1.transformer_blocks.2.attn2.processor",
                           "up_blocks.0.attentions.1.transformer_blocks.3.attn2.processor",
                           "up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor",
                           "up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor",
                           "up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor"]
    attn_procs = {}
    total_hooked_layers = 0
    tome_hooked_layers = 0

    for name, processor in model.unet.attn_processors.items():
        place_in_unet = None
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            attn_procs[name] = processor
            continue
        
        tome_control_point = False
        if name in tome_attn_greenlist:
            tome_control_point = True
            tome_hooked_layers += 1
        attn_procs[name] = CombinedAttentionProcessor(
            tome_controller=tome_controller,
            self_attn_controller=self_attn_controller,
            place_in_unet=place_in_unet,
            tome_control_point=tome_control_point,
        )
        total_hooked_layers += 1


    model.unet.set_attn_processor(attn_procs)

    tome_controller.num_att_layers = tome_hooked_layers
    self_attn_controller.num_att_layers = total_hooked_layers
    
    return attn_procs, total_hooked_layers