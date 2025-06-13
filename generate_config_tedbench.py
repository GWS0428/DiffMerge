import os
import json
import numpy as np
from PIL import Image
from datasets import load_dataset


def count_words(prompt):
    words = prompt.strip().split()
    return len(words)


# TedBench (imagic)
ds = load_dataset("bahjat-kawar/tedbench")
        
# generate a config file for each image
defualt_config = json.load(open("configs/config_default.json", 'r'))
for i, image in enumerate(ds['val']['original_image']):
    if isinstance(image, Image.Image):
        image.save(f"datasets/tedbench/image_{i:03}.png")
        print(f"Saved image {i:03} to datasets/tedbench/image_{i:03}.png")
        
        # create a config file for each image
        config = defualt_config.copy()
        config['image_path'] = f"datasets/tedbench/image_{i:03}.png"
        config['prompt'] = ds['val']['caption'][i].rstrip('.')
        config['token_indices'] = []
        config['prompt_anchor'] = []
        config['prompt_merged'] = ds['val']['caption'][i].rstrip('.')
        config['prompt_length'] = count_words(ds['val']['caption'][i])
        
        # file name as prompt
        prompt_filename = ds['val']['caption'][i].replace(" ", "_").replace("/", "_").replace("\\", "_")
        config['output_path'] = f"./demo/tedbench"
        
        with open(f"configs/tedbench/{prompt_filename}.json", 'w') as f:
            json.dump(config, f, indent=4)
            
        print(f"Saved config for image {i:03} with prompt '{ds['val']['caption'][i]}'")
    else:
        raise TypeError(f"Expected PIL Image, got {type(image)}")
