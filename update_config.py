import os
import re
import json
import argparse

import numpy as np


def update_config(
    config_dir, 
    run_standard_sd=False,
    tome_control_steps=[0, 0],
    token_refinement_steps=0,
    attention_refinement_steps=[0, 0],
    eot_replace_step=0,
    self_replace_steps=0.6,
    ):
    # iterate through all JSON files in the config directory
    for filename in os.listdir(config_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(config_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                if 'run_standard_sd' in config:
                    config['run_standard_sd'] = run_standard_sd
                if 'tome_control_steps' in config:
                    config['tome_control_steps'] = tome_control_steps
                if 'token_refinement_steps' in config:
                    config['token_refinement_steps'] = token_refinement_steps
                if 'attention_refinement_steps' in config:
                    config['attention_refinement_steps'] = attention_refinement_steps
                if 'eot_replace_step' in config:
                    config['eot_replace_step'] = eot_replace_step
                if 'self_replace_steps' in config:
                    config['self_replace_steps'] = self_replace_steps
                
                # Save the updated config back to the file
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=4)
                
                print(f"Updated config file: {file_path}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error processing {file_path}: {e}")
                exit(1)
                

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Update configuration files in a directory.")
    argparser.add_argument(
        "--config_dir", type=str, required=True, help="Directory containing the config files."
    )
    argparser.add_argument(
        "--run_standard_sd", action='store_true', help="Set to run standard SD."
    )
    argparser.add_argument(
        "--tome_control_steps", type=int, nargs=2, default=[2, 2], help="TOME control steps."
    )
    argparser.add_argument(
        "--token_refinement_steps", type=int, default=2, help="Token refinement steps."
    )
    argparser.add_argument(
        "--attention_refinement_steps", type=int, nargs=2, default=[2, 2], help="Attention refinement steps."
    )
    argparser.add_argument(
        "--eot_replace_step", type=int, default=20, help="EOT replace step."
    )
    argparser.add_argument(
        "--self_replace_steps", type=float, default=0.6, help="Self replace steps."
    )
    
    args = argparser.parse_args()
    update_config(
        config_dir=args.config_dir,
        run_standard_sd=args.run_standard_sd,
        tome_control_steps=args.tome_control_steps,
        token_refinement_steps=args.token_refinement_steps,
        attention_refinement_steps=args.attention_refinement_steps,
        eot_replace_step=args.eot_replace_step,
        self_replace_steps=args.self_replace_steps
    )