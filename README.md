
# 🌟 EditMerge: Rethinking Token Merging for Semantic Binding in Diffusion-based Image Editing

<!-- <img src="pics/teaser.png" width="1000"> -->

## 📑 Introduction

This project explores the opportunity to use token merging as a semantic binding tool in diffusion based image editing. **Semantic binding** is defined as the task of associating an object with its attribute (attribute binding) or linking it to related sub-objects (object binding). We enhanced semantic binding by aggregating relevant tokens into a single composite token, aligning the object, its attributes, and sub-objects in the same cross-attention map.

For technical details, please refer to our final ppt slides.

## 🚀 Usage

1. **Environment Setup**

   **Create and activate the Conda virtual environment:**

   ```bash
   conda create -n editmerge python=3.12 -y
   conda activate editmerge
   pip install -r requirements.txt
   ```

2. **Configure Parameters**

   The default config file is given as `configs/config_default.json`. Modify this file to adjust runtime parameters as needed. Key parameters are as follows:

   - `prompt`: Text prompt for guiding image generation.
   - `model_path`: Path to the Stable Diffusion model; set to `None` to download the pretrained model automatically.
   - `token_indices`: Indices of tokens to merge.
   - `prompt_anchor`: Split text prompt.
   - `prompt_merged`: Text prompt after token merging.
   - `prompt_length`: Text prompt length after token merging.

   <!-- - `fpe_ratio`:  -->
   
   For further parameter details, please refer to the comments in the configuration file.

3. **Run the Example**

   Execute the main script `run_demo.py`:

   ```bash
   python run_demo.py
   ```

   The generated images will be saved in the `demo` directory.

4. **Benchmarks**

   We tested our method on ImagenetR-ti2i, Wild-real-ti2i, TedBench datasets. You can download the datasets from the official PnP website. [Link](https://github.com/MichalGeyer/plug-and-play)

   <!-- Steps to run a benchmark are as follows
   - `aaa` -->

## 📸 Example Outputs

If everything is set up correctly, `configs/config_default.json` should produce the image below:

<!-- <img src="pics\demo.png" width="1000"> -->

## ⚠️ Notes

- **Custom Configurations**: To use custom text prompts and parameters, add a new configuration in `configs/config_default.py` and make necessary adjustments in `run_demo.py`.
- **Parameter Sensitivity**: This method inherits the sensitivity of inference-based optimization techniques, meaning that the generated results are highly dependent on hyperparameter settings. Careful tuning may be required to achieve optimal results.

## 🙏 Acknowledgments

This project builds upon valuable work from the following repositories:

- [ToMe](https://github.com/hutaihang/ToMe) 
- [Diffusers 🤗](https://github.com/huggingface/diffusers) 

We extend our sincere thanks to the creators of these projects for their contributions to the field and for making their code available. 🙌

