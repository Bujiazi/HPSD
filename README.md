<div align="center">

<p align="center">
  <img src="__assets__/hpsd_logo.png" height=150>
</p>

<h1>HPSD: Hybrid-Policy Self-Distillation for Text-Image-to-Video Diffusion Models</h1>

<div>
    <a href="https://scholar.google.com/citations?user=a8h9Di4AAAAJ" target="_blank">Jiazi Bu*</a><sup></sup> | 
    <a href="https://github.com/LPengYang/" target="_blank">Pengyang Ling*<sup>§</sup></a><sup></sup> | 
    <a href="https://github.com/YujieOuO" target="_blank">Yujie Zhou*</a><sup></sup> |
    <a href="https://codegoat24.github.io/" target="_blank">Yibin Wang</a><sup></sup> |
    <a href="https://yuhangzang.github.io/" target="_blank">Yuhang Zang</a><sup></sup> |
    <a href="https://scholar.google.com/citations?user=OeeH1HAAAAAJ&hl=en" target="_blank">Xuanlang Dai</a><sup></sup> | <br>
    <a href="https://scholar.google.com/citations?user=iDPJVBsAAAAJ&hl=zh-CN" target="_blank">Shengyuan Ding</a><sup></sup> | 
    <a href="https://wtybest.github.io/" target="_blank">Tianyi Wei</a><sup></sup> |
    <a href="https://xiaohangzhan.github.io/" target="_blank">Xiaohang Zhan</a><sup></sup> |
    <a href="https://myownskyw7.github.io/" target="_blank">Jiaqi Wang</a><sup></sup> |
    <a href="https://wutong16.github.io/" target="_blank">Tong Wu</a><sup></sup> |
    <a href="http://dahua.site/" target="_blank">Dahua Lin</a><sup></sup> |
    <a href="https://xingangpan.github.io/" target="_blank">Xingang Pan<sup>†</sup></a><sup></sup>
</div>
<br>
<div>
    <sup></sup>Shanghai Jiao Tong University, Nanyang Technological University, Shanghai AI Laboratory, 
    <br> University of Science and Technology of China, Fudan University, Shanghai Innovation Institute
    <br> The Chinese University of Hong Kong, CPII under InnoHK, JD.com, Adobe Research
</div>
(*<b>Equal Contribution</b>)(<sup>§</sup><b>Project Leader</b>)(<sup>†</sup><b>Corresponding Author</b>)
<br><br>


[![arXiv](https://img.shields.io/badge/arXiv-2608.13205-b31b1b.svg)](https://arxiv.org/abs/2608.13205) 
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://bujiazi.github.io/hpsd.github.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-red)](https://huggingface.co/Bujiazi/HPSD)


---

<strong>HPSD is a hybrid-policy self-distillation framework for improving the base generation ability of TI2V models.</strong>

<details><summary>📖 Click for the full abstract of HPSD</summary>

<div align="left">

> Text-Image-to-Video (TI2V) models are an emerging unified architecture, where a single model simultaneously supports text-to-video (T2V) and image-to-video (I2V) generation. Given a high-quality first frame or a detailed textual prompt, TI2V models unlock substantially better visual quality than their T2V mode, raising a natural question: _can the capability elicited by such privileged conditions be internalized into the model's own base generation ability?_ A common approach toward this goal is model self-distillation. However, the most straightforward solution, supervised fine-tuning, follows an off-policy strategy: its supervision is confined to teacher-generated endpoints from a fixed offline distribution rather than student-visited states, lacking precise correction tailored to the evolving policy. Recent on-policy distillation methods instead suffer from condition-state mismatch, where supervision is steered toward the given first frame instead of the student's actual content, misleading the correction. To achieve self-distillation that absorbs the teacher's privileged prior while retaining precise policy correction, in this work, we propose **H**ybrid-**P**olicy **S**elf-**D**istillation (**HPSD**), a novel self-distillation framework where a single TI2V model acts as both teacher and student under different conditions: the teacher operates in TI2V mode with a high-quality first frame and an enhanced prompt, while the student runs in the base T2V mode with only the vanilla prompt. Specifically, the student inherits off-policy teacher trajectory points as anchors, locally refines them toward its own policy, and finally receives velocity-level supervision on these self-generated roll-outs. Extensive experiments demonstrate that HPSD significantly improves T2V performance while also delivering notable TI2V gains, effectively strengthening the model's base generation ability.
</details>
</div>

## 🎉 Quick Demos
More results can be found in the [Project Website](https://bujiazi.github.io/hpsd.github.io/).

<table class="center">
    <tr>
    <td style="text-align:center;"><b>Vanilla T2V</b></td>
    <td style="text-align:center;"><b>HPSD (Ours)</b></td>
    </tr>
    <tr>
    <td><img src="__assets__/videos/1_vanilla.gif"></td>
    <td><img src="__assets__/videos/1_hpsd.gif"></td>
    </tr>
    <tr>
    <td colspan="2" style="text-align:center;"><i>"a ship sailing in the ocean."</i></td>
    </tr>
    <tr>
    <td><img src="__assets__/videos/4_vanilla.gif"></td>
    <td><img src="__assets__/videos/4_hpsd.gif"></td>
    </tr>
    <tr>
    <td colspan="2" style="text-align:center;"><i>"cars doing a super race in New York."</i></td>
    </tr>
    <tr>
    <td><img src="__assets__/videos/5_vanilla.gif"></td>
    <td><img src="__assets__/videos/5_hpsd.gif"></td>
    </tr>
    <tr>
    <td colspan="2" style="text-align:center;"><i>"a long eared cat with stripes walking and blinking towards the camera on a hill"</i></td>
    </tr>
    <tr>
    <td><img src="__assets__/videos/6_vanilla.gif"></td>
    <td><img src="__assets__/videos/6_hpsd.gif"></td>
    </tr>
    <tr>
    <td colspan="2" style="text-align:center;"><i>"bar scene cinematic where a handsome young man with blue eyes is walking."</i></td>
    </tr>
    <tr>
    <td><img src="__assets__/videos/2_vanilla.gif"></td>
    <td><img src="__assets__/videos/2_hpsd.gif"></td>
    </tr>
    <tr>
    <td colspan="2" style="text-align:center;"><i>"Drone flyover of a Jovian sky colony."</i></td>
    </tr>
    <tr>
    <td><img src="__assets__/videos/3_vanilla.gif"></td>
    <td><img src="__assets__/videos/3_hpsd.gif"></td>
    </tr>
    <tr>
    <td colspan="2" style="text-align:center;"><i>"photo of coastline, rocks, distant lighthouse in the background, storm weather, strong wind, crashing waves, ..."</i></td>
    </tr>
</table>


## 🎨 Overview
<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="__assets__/hpsd_teaser.png">
</div>
<br>

## 💻 Method
<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="__assets__/hpsd_pipeline.png">
</div>
<br>

HPSD lets the student start from the teacher's trajectory and finish under its own policy, which proceeds in the following stages: **(a) Offline Stage**: Before training, the privileged conditions are synthesized for each prompt with off-the-shelf generative models. **(b) Online Stage**: During training, the teacher first rolls out its full denoising trajectory under privileged conditions, yielding an off-policy anchor trajectory that carries the condition-elicited generation content; The student then continues denoising from intermediate states of the anchor trajectory with its own velocity field, and is supervised by the teacher on the resulting sub-trajectory. Since the supervised states are anchored by the teacher's policy yet evolved by the student's, we term this design a _hybrid-policy_.

</details>
</div>

## 🖋 News
- The first version of HPSD code has been released! (2026.8.22)
- Pre-trained checkpoint is available on Huggingface! (2026.8.22)
- Our Project page is available now! (2026.8.14)
- Our Paper has been released on arXiv! (2026.8.14)

## 🔧 Installations

### Setup repository and conda environment

```bash
git clone https://github.com/Bujiazi/HPSD.git
cd HPSD

conda create -n hpsd python=3.10
conda activate hpsd

pip install -r requirements.txt
```

It is strongly recommended to install the pre-compiled `flash_attn`.

### Download pretrained models

```bash
bash scripts/download_pretrained_models.sh
```

Alternatively, you can download the checkpoints manually and place them following the layout below:

```
pretrained_models/
├── WAN2.2-TI2V/                          # Wan-AI/Wan2.2-TI2V-5B-Diffusers
│   ├── transformer/                      #   WanTransformer3DModel
│   ├── vae/                              #   AutoencoderKLWan
│   ├── text_encoder/                     #   UMT5EncoderModel
│   ├── tokenizer/                        #   T5TokenizerFast
│   ├── scheduler/                        #   UniPCMultistepScheduler
│   └── model_index.json                  #   WanPipeline
├── Z-Image-Turbo/                        # Tongyi-MAI/Z-Image-Turbo
│   ├── transformer/                      #   ZImageTransformer2DModel
│   ├── vae/                              #   AutoencoderKL
│   ├── text_encoder/                     #   Qwen3Model
│   ├── tokenizer/                        #   Qwen2Tokenizer
│   ├── scheduler/                        #   FlowMatchEulerDiscreteScheduler
│   └── model_index.json                  #   ZImagePipeline
└── Qwen3.6-27B/                          # Qwen/Qwen3.6-27B
    ├── config.json
    ├── generation_config.json
    ├── model-00001-of-00015.safetensors  
    ├── ...
    ├── model-00015-of-00015.safetensors
    ├── model.safetensors.index.json
    ├── tokenizer.json                    
    └── chat_template.jinja
```

## 🎈 Quick Start

### Data Preparation

```bash
bash scripts/prepare_data.sh
```

### HPSD Training

```bash
bash scripts/train_hpsd.sh
```

### HPSD Inference

Try our pretrained HPSD checkpoint on [Huggingface🤗](https://huggingface.co/Bujiazi/HPSD).

```python
import torch
from huggingface_hub import snapshot_download
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from peft import PeftModel

dtype = torch.bfloat16
device = "cuda"

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
checkpoint = snapshot_download(repo_id="Bujiazi/HPSD", repo_type="model")

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
pipe.to(device)

pipe.transformer = PeftModel.from_pretrained(pipe.transformer, checkpoint, torch_dtype=dtype).to(device)

height = 704
width = 1280
num_frames = 81
num_inference_steps = 50
guidance_scale = 5.0
base_seed = 42

prompt = "A white horse galloping across an open grassland under dramatic clouds. Its mane flows naturally in the wind, cinematic wide shot, realistic movement, soft sunlight."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

generator = torch.Generator(device=device).manual_seed(base_seed)
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=generator,
).frames[0]
export_to_video(output, "hpsd_test.mp4", fps=16)
```


## 🏗️ Todo
- [x] 🚀 Release checkpoint on Huggingface
- [x] 🚀 Release HPSD code
- [x] 🚀 Release the project page
- [x] 🚀 Release paper

## 📎 Citation 

If you find our work helpful, please consider giving a star ⭐ and citation 📝 
```bibtex
@article{bu2026hpsd,
  title={HPSD: Hybrid-Policy Self-Distillation for Text-Image-to-Video Diffusion Models},
  author={Bu, Jiazi and Ling, Pengyang and Zhou, Yujie and Wang, Yibin and Zang, Yuhang and Dai, Xuanlang and Ding, Shengyuan and Wei, Tianyi and Zhan, Xiaohang and Wang, Jiaqi and others},
  journal={arXiv preprint arXiv:2608.13205},
  year={2026}
}
```

## 📣 Disclaimer

This is official code of HPSD.
All the copyrights of the demo images and audio are from community users. 
Feel free to contact us if you would like remove them.

## 💞 Acknowledgements
The code is built upon the below repositories, we thank all the contributors for open-sourcing.
* [Diffusers](https://github.com/huggingface/diffusers)
* [WAN-2.2](https://github.com/Wan-Video/Wan2.2)
* [LTX-2.3](https://github.com/Lightricks/LTX-2)
* [Z-Image](https://github.com/Tongyi-MAI/Z-Image)
* [Qwen3.6-27B](https://github.com/QwenLM/Qwen3.6)
* [D-OPSD](https://github.com/vvvvvjdy/D-OPSD)
