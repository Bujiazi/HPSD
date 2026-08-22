from __future__ import annotations

from tqdm.auto import tqdm

from arguments import parse_args
from dataset import VideoPromptDataset
from first_frame_utils import FirstFrameManager


def main() -> None:
    args = parse_args(require_exp_name=False)
    dataset = VideoPromptDataset(
        args.prompt_file,
        prompt_key=args.prompt_key,
        max_samples=args.max_train_samples,
        start=args.prompt_start,
        shuffle=args.shuffle_prompts,
        shuffle_seed=args.prompt_shuffle_seed,
    )
    print(
        f"Preparing first-frame + enhanced-prompt cache for {len(dataset)} prompts "
        f"(slice [{args.prompt_start}, {args.prompt_start + len(dataset)}))"
    )

    manager = FirstFrameManager(
        cache_dir=args.first_frame_cache_dir,
        qwen_model_path=args.qwen_model_path,
        zimage_model_path=args.zimage_model_path,
        template_path=args.first_frame_template,
        placeholder=args.first_frame_placeholder,
        height=args.height,
        width=args.width,
        zimage_steps=args.zimage_steps,
        zimage_guidance_scale=args.zimage_guidance_scale,
        base_seed=args.first_frame_seed,
    )

    try:
        for i in tqdm(range(len(dataset)), desc="HPSD data cache"):
            item = dataset[i]
            manager.ensure(item["prompt"], item["id"])
    finally:
        manager.close()


if __name__ == "__main__":
    main()
