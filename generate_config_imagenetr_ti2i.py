import os
import re
import json
import yaml


# imagenetr-ti2i (PnP)
print("Loading benchmark data for imagenetr-ti2i...")
# wget -O dataset.zip "https://www.dropbox.com/scl/fo/dzda33xyvng57armyoygb/AHH9oiuzr1-4rtJdLKmvwWA?rlkey=a5bn2vttu4zllu4qv8lodt6ie&dl=1"


## generate config files automatically

# Define the path to the default config JSON file & dataset 
# current file path
file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)
default_config_file_path = f"{file_dir}/configs/config_default.json"
yaml_file_path = f"{file_dir}/datasets/imagenetr-ti2i/imnetr-ti2i.yaml"
output_dir = f"{file_dir}/configs/imagenetr-ti2i"
os.makedirs(output_dir, exist_ok=True) 

# Load the default configuration from the JSON file
try:
    with open(default_config_file_path, 'r') as f:
        default_config = json.load(f)
    print(f"Successfully loaded default config from {default_config_file_path}")
except FileNotFoundError:
    print(f"Error: The default config file '{default_config_file_path}' was not found. Please ensure it exists.")
    exit()
except json.JSONDecodeError as e:
    print(f"Error parsing default config JSON file '{default_config_file_path}': {e}")
    exit()

# Load the benchmark data from the YAML file
try:
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    print(f"Successfully loaded benchmark data from {yaml_file_path}")
except FileNotFoundError:
    print(f"Error: The benchmark data file '{yaml_file_path}' was not found. Please ensure it exists in the same directory.")
    exit()
except yaml.YAMLError as e:
    print(f"Error parsing YAML file '{yaml_file_path}': {e}")
    exit()
    
def count_words(prompt):
    words = prompt.strip().split()
    return len(words)

# Helper function to sanitize text for use in a filename
def sanitize_filename(text, max_len=50):
    """
    Sanitizes a string to be suitable for a filename by replacing spaces with
    underscores, removing special characters, and truncating if too long.
    """
    # Replace spaces with underscores
    s = text.replace(" ", "_")
    # Remove characters that are not alphanumeric, underscore, or hyphen
    s = re.sub(r"[^\w.-]", "", s)
    s = s.lower()

    # Truncate if too long
    if len(s) > max_len:
        truncated_s = s[:max_len]
        last_underscore_idx = truncated_s.rfind('_')
        if last_underscore_idx != -1 and last_underscore_idx > max_len * 0.5:
            s = truncated_s[:last_underscore_idx]
        else:
            s = truncated_s

    return s

# Iterate through each entry in the parsed YAML data
for entry in yaml_data:
    init_img = entry["init_img"]
    target_prompts = entry["target_prompts"]

    # Extract the base name of the initial image (e.g., "real_karate")
    img_name_base = os.path.splitext(os.path.basename(init_img))[0]
    init_img_dir = os.path.dirname(init_img)
    init_img_dir = os.path.basename(init_img_dir)
    img_name_pure = img_name_base.split('_')[0]  # Get the first part before the underscore

    # For each target prompt associated with the current initial image
    for i, prompt in enumerate(target_prompts):
        # Create a shallow copy of the default config dictionary
        current_config = default_config.copy()

        # Update the 'img_path' and 'prompt' fields
        current_config["img_path"] = init_img
        current_config["prompt"] = prompt
        current_config["src_prompt"] = f"a photo of a {init_img_dir} {img_name_pure}"
        current_config["output_path"] = f"./demo/imagenetr-ti2i"

        # Fill the tome relevant fields with the trivial values
        current_config["token_indices"] = []
        current_config["prompt_anchor"] = []
        current_config["prompt_merged"] = prompt
        current_config["prompt_length"] = count_words(prompt)
        
        # Generate a unique filename for the new JSON config file
        sanitized_prompt_name = sanitize_filename(prompt)
        filename = f"{img_name_base}_{sanitized_prompt_name}_{i}.json"
        file_path = os.path.join(output_dir, filename)

        # Save the updated config to a JSON file
        with open(file_path, "w") as f:
            json.dump(current_config, f, indent=2) # Use indent=2 for pretty-printing JSON

        print(f"Generated config file: {file_path}")

print(f"\nAll configuration files have been successfully generated in the '{output_dir}' directory.")
print("dataset: imagenetr-ti2i")