import os
import shutil
import itertools
import subprocess
from pathlib import Path

# Define all options
run_standard_sd_options = [True, False]
tome_control_steps_options = [[1, 1], [2, 2], [3, 3]]
# tome_control_steps_options = [[1, 1], [2, 2]]
token_refinement_steps_options = [1, 2, 3]
# attention_refinement_steps_options = [[1, 1], [2, 2], [3, 3]]
attention_refinement_steps_options = [[1, 1], [2, 2]]
eot_replace_step_options = [0, 20, 60]
self_replace_steps_options = [0.2, 0.3, 0.4, 0.5]

def format_value(val):
    if isinstance(val, list):
        return ''.join(str(x) for x in val)
    elif isinstance(val, float):
        return f"{int(val * 10)}"
    else:
        return str(val)

def generate_new_dir_name(base_name, params):
    parts = [
        f"run_standard_sd-{str(params['run_standard_sd']).lower()}",
        f"tome_control_steps-{format_value(params['tome_control_steps'])}",
        f"token_refinement_steps-{format_value(params['token_refinement_steps'])}",
        f"attention_refinement_steps-{format_value(params['attention_refinement_steps'])}",
        f"eot_replace_step-{format_value(params['eot_replace_step'])}",
        f"self_replace_steps-{format_value(params['self_replace_steps'])}"
    ]
    return f"{base_name}_{'_'.join(parts)}"

def main():
    cur_file_dir_path = Path(__file__).parent.resolve()
    original_config_dir = cur_file_dir_path / "configs" / "wild-ti2i"
    update_script = "update_config.py"

    base_name = Path(original_config_dir).name
    parent_dir = Path(original_config_dir).parent

    for run_standard_sd in run_standard_sd_options:
        if run_standard_sd:
            param_grid = itertools.product(self_replace_steps_options)
        else:
            param_grid = itertools.product(
                tome_control_steps_options,
                token_refinement_steps_options,
                attention_refinement_steps_options,
                eot_replace_step_options,
                self_replace_steps_options
            )

        for params in param_grid:
            if run_standard_sd:
                param_dict = {
                    "run_standard_sd": True,
                    "tome_control_steps": [0, 0],
                    "token_refinement_steps": 0,
                    "attention_refinement_steps": [0, 0],
                    "eot_replace_step": 0,
                    "self_replace_steps": params[0],
                }
            else:
                param_dict = {
                    "run_standard_sd": False,
                    "tome_control_steps": params[0],
                    "token_refinement_steps": params[1],
                    "attention_refinement_steps": params[2],
                    "eot_replace_step": params[3],
                    "self_replace_steps": params[4],
                }

            new_dir_name = generate_new_dir_name(base_name, param_dict)
            new_dir_path = parent_dir / new_dir_name
            output_path = f"./demo/{new_dir_name}"

            # Ensure clean directory copy
            if new_dir_path.exists():
                shutil.rmtree(new_dir_path)
            shutil.copytree(original_config_dir, new_dir_path)

            # Build command
            cmd = [
                "python", update_script,
                "--config_dir", str(new_dir_path),
                "--tome_control_steps", *map(str, param_dict["tome_control_steps"]),
                "--token_refinement_steps", str(param_dict["token_refinement_steps"]),
                "--attention_refinement_steps", *map(str, param_dict["attention_refinement_steps"]),
                "--eot_replace_step", str(param_dict["eot_replace_step"]),
                "--self_replace_steps", str(param_dict["self_replace_steps"]),
                "--output_path", str(output_path),
            ]

            if param_dict["run_standard_sd"]:
                cmd.append("--run_standard_sd")

            # Run the command
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
