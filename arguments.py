import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_int_list(s: str | None) -> list[int] | None:
    if s is None or s.strip() == "":
        return None
    return [int(x) for x in s.split()]


DEFAULT_MODEL_ID = str(PROJECT_ROOT / "pretrained_models" / "WAN2.2-TI2V")
DEFAULT_PROMPT_FILE = str(PROJECT_ROOT / "data" / "video_prompts.txt")
DEFAULT_QWEN_PATH = str(PROJECT_ROOT / "pretrained_models" / "Qwen3.6-27B")
DEFAULT_ZIMAGE_PATH = str(PROJECT_ROOT / "pretrained_models" / "Z-Image-Turbo")
DEFAULT_TEMPLATE = str(PROJECT_ROOT / "data" / "prompt_template.txt")
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args(require_exp_name: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WAN2.2 TI2V-teacher -> T2V-student HPSD")

    parser.add_argument("--deepspeed-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "exp_results"))
    parser.add_argument("--exp-name", type=str, default=None if require_exp_name else "prepare_data", required=require_exp_name)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wan-model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--teacher-inference-steps", type=int, default=50)
    parser.add_argument("--teacher-guidance-scale", type=float, default=5.0)
    parser.add_argument("--teacher-guidance-scale-2", type=float, default=None)
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--max-sequence-length", type=int, default=512)

    parser.add_argument("--prompt-file", type=str, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--prompt-key", type=str, default="prompt")
    parser.add_argument("--max-train-samples", type=int, default=5000, help="Number of training prompts to use.")
    parser.add_argument("--prompt-start", type=int, default=0, help="Skip the first N valid prompts.")
    parser.add_argument("--shuffle-prompts", action=argparse.BooleanOptionalAction, default=False, help="Sample prompts uniformly at random from the whole prompt file.")
    parser.add_argument("--prompt-shuffle-seed", type=int, default=42, help="Seed for --shuffle-prompts.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--precompute-first-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--first-frame-cache-dir", type=str, default=str(PROJECT_ROOT / "cached_data"))
    parser.add_argument("--first-frame-template", type=str, default=DEFAULT_TEMPLATE)
    parser.add_argument("--first-frame-placeholder", type=str, default="{请在这里填入你的视频Prompt}")
    parser.add_argument("--qwen-model-path", type=str, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--zimage-model-path", type=str, default=DEFAULT_ZIMAGE_PATH)
    parser.add_argument("--zimage-steps", type=int, default=9)
    parser.add_argument("--zimage-guidance-scale", type=float, default=0.0)
    parser.add_argument("--first-frame-seed", type=int, default=42)

    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-target-modules", type=str, default="to_q,to_k,to_v,to_out.0,add_k_proj,add_v_proj,ffn.net.0.proj,ffn.net.2")
    parser.add_argument("--student-adapter", type=str, default="student")
    parser.add_argument("--teacher-adapter", type=str, default="teacher")
    parser.add_argument("--ema-decay", type=float, default=0.9999)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-train-steps", type=int, default=2000)
    parser.add_argument("--traj-high-steps", type=int, default=3, help="Number of steps randomly sampled from the high-noise region of the teacher trajectory.")
    parser.add_argument("--traj-low-steps", type=int, default=3, help="Number of steps randomly sampled from the low-noise region of the teacher trajectory.")
    parser.add_argument("--inference-steps", type=int, default=None, help="Denoising steps for both teacher and student trajectories.")
    parser.add_argument("--anchor-indices", type=parse_int_list, default=None, help="Explicit teacher-trajectory step indices to collect.")
    parser.add_argument("--sub-traj-length-list", type=parse_int_list, default=None, help="Per-anchor-index student continuation steps K for HPSD.")
    parser.add_argument("--hybrid-continue-steps", type=int, default=3, help="Number of student continuation denoising steps (K) rolled out from the teacher anchor point before HPSD alignment.")
    parser.add_argument("--train-guidance-scale", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--transformer-dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--vae-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--enable-gc", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--save-sample-every-steps", type=int, default=10, help="Save teacher/student sample videos every N optimizer steps; set 0 to disable.")
    parser.add_argument("--sample-inference-steps", type=int, default=None, help="Student sample denoising steps; defaults to --teacher-inference-steps.")
    parser.add_argument("--sample-guidance-scale", type=float, default=5.0)
    parser.add_argument("--save-sample-fps", type=int, default=16)

    return parser.parse_args()
