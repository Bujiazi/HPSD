from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import gc
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import is_compiled_module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from arguments import parse_args
from dataset import VideoPromptDataset, collate_prompts
from first_frame_utils import FirstFrameManager
from lora_utils import ema_update_lora_adapter, init_dual_lora_model, module_of, set_active_adapter, trainable_parameters


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def unwrap_model(model, accelerator: Accelerator):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def make_logger(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(save_dir, "log.txt"))],
    )
    return logging.getLogger(__name__)


def retrieve_latents(encoder_output, sample_mode: str = "argmax") -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        if sample_mode == "sample":
            return encoder_output.latent_dist.sample()
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not retrieve latents from VAE encoder output")


def vae_latent_stats(vae, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    return mean, std


@torch.no_grad()
def prepare_ti2v_condition(
    pipe: WanImageToVideoPipeline,
    images: list[Any],
    latents: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    image = pipe.video_processor.preprocess(images, height=height, width=width).to(device=device, dtype=torch.float32)
    bsz, _, latent_frames, latent_h, latent_w = latents.shape

    if pipe.config.expand_timesteps:
        video_condition = image.unsqueeze(2)
    else:
        video_condition = torch.cat(
            [image.unsqueeze(2), image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
        )
    video_condition = video_condition.to(device=device, dtype=pipe.vae.dtype)

    latent_condition = retrieve_latents(pipe.vae.encode(video_condition), sample_mode="argmax")
    latent_condition = latent_condition.to(device=device, dtype=latents.dtype)
    latents_mean, latents_std = vae_latent_stats(pipe.vae, latents.device, latents.dtype)
    latent_condition = (latent_condition - latents_mean) * latents_std

    image_embeds = None
    model_for_config = pipe.transformer if pipe.transformer is not None else pipe.transformer_2
    model_cfg = getattr(module_of(model_for_config), "config", None) if model_for_config is not None else None
    if model_cfg is not None and getattr(model_cfg, "image_dim", None) is not None:
        image_embeds = pipe.encode_image(images, device=device).repeat(bsz, 1, 1)
        image_embeds = image_embeds.to(device=device, dtype=get_model_dtype(model_for_config))

    if pipe.config.expand_timesteps:
        first_frame_mask = torch.ones(bsz, 1, latent_frames, latent_h, latent_w, dtype=latents.dtype, device=device)
        first_frame_mask[:, :, 0] = 0
        return latent_condition, first_frame_mask, image_embeds

    mask_lat_size = torch.ones(bsz, 1, num_frames, latent_h, latent_w, dtype=latents.dtype, device=device)
    mask_lat_size[:, :, list(range(1, num_frames))] = 0
    first_frame_mask = mask_lat_size[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=pipe.vae_scale_factor_temporal)
    mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
    mask_lat_size = mask_lat_size.view(bsz, -1, pipe.vae_scale_factor_temporal, latent_h, latent_w)
    mask_lat_size = mask_lat_size.transpose(1, 2).to(device=device, dtype=latents.dtype)
    return torch.concat([mask_lat_size, latent_condition], dim=1), None, image_embeds


def initial_latents(pipe: WanImageToVideoPipeline, batch_size: int, height: int, width: int, num_frames: int, device: torch.device, seed: int | None = None) -> torch.Tensor:
    latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    latent_h = height // pipe.vae_scale_factor_spatial
    latent_w = width // pipe.vae_scale_factor_spatial
    generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    return torch.randn((batch_size, pipe.vae.config.z_dim, latent_frames, latent_h, latent_w), generator=generator, device=device, dtype=torch.float32)


def choose_model(pipe: WanImageToVideoPipeline, t: torch.Tensor):
    if pipe.config.boundary_ratio is None or pipe.transformer_2 is None:
        return pipe.transformer, None
    boundary_timestep = pipe.config.boundary_ratio * pipe.scheduler.config.num_train_timesteps
    if float(t.item()) >= float(boundary_timestep):
        return pipe.transformer, "high"
    return pipe.transformer_2, "low"


def timestep_for(pipe: WanImageToVideoPipeline, latents: torch.Tensor, t: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if pipe.config.expand_timesteps:
        if mask is None:
            mask = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4], device=latents.device, dtype=latents.dtype)
        temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
        return temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
    return t.expand(latents.shape[0])


def get_model_dtype(model) -> torch.dtype:
    return next(module_of(model).parameters()).dtype


def model_cache_context(model, name: str):
    for candidate in (model, module_of(model), getattr(module_of(model), "base_model", None)):
        if candidate is not None and hasattr(candidate, "cache_context"):
            return candidate.cache_context(name)
    return contextlib.nullcontext()


@contextlib.contextmanager
def only_active_adapter(model, adapter_name: str):
    if model is None:
        yield
        return
    target = module_of(model)
    if hasattr(target, "set_adapter"):
        target.set_adapter(adapter_name)
    yield


@contextlib.contextmanager
def eval_mode(*models):
    states = []
    try:
        for model in models:
            if model is None:
                continue
            target = module_of(model)
            states.append((target, target.training))
            target.eval()
        yield
    finally:
        for target, train_state in states:
            target.train(train_state)


def forward_velocity(
    model,
    adapter: str | None,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    cache_name: str = "cond",
) -> torch.Tensor:
    if adapter is not None:
        set_active_adapter(model, adapter)
    model_dtype = get_model_dtype(model)
    kwargs = {"attention_kwargs": None}
    if encoder_hidden_states_image is not None:
        kwargs["encoder_hidden_states_image"] = encoder_hidden_states_image.to(dtype=model_dtype)
    with model_cache_context(model, cache_name):
        return model(
            hidden_states=hidden_states.to(dtype=model_dtype),
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states.to(dtype=model_dtype),
            return_dict=False,
            **kwargs,
        )[0].float()


def guided_velocity(
    model,
    adapter: str | None,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    guidance_scale: float,
    image_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = forward_velocity(model, adapter, hidden_states, timestep, prompt_embeds, image_embeds, cache_name="cond")
    if negative_prompt_embeds is None or guidance_scale <= 1.0:
        return pred
    uncond = forward_velocity(model, adapter, hidden_states, timestep, negative_prompt_embeds, image_embeds, cache_name="uncond")
    return uncond + guidance_scale * (pred - uncond)


@torch.no_grad()
def decode_latents_to_frames(pipe: WanImageToVideoPipeline, latents: torch.Tensor, output_type: str = "pil"):
    latents = latents.to(pipe.vae.dtype)
    latents_mean, latents_std = vae_latent_stats(pipe.vae, latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type=output_type)


@torch.no_grad()
def generate_t2v_student_latents(
    pipe: WanImageToVideoPipeline,
    prompt: str,
    args,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    num_inference_steps = args.sample_inference_steps or args.teacher_inference_steps
    do_cfg = args.sample_guidance_scale > 1.0
    with eval_mode(pipe.transformer, pipe.transformer_2):
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=[prompt],
            negative_prompt=[args.negative_prompt],
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(device=device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device)

        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = initial_latents(pipe, 1, args.height, args.width, args.num_frames, device, seed=seed)
        for t in pipe.scheduler.timesteps:
            t = t.to(device=device, dtype=torch.float32)
            current_model, _ = choose_model(pipe, t)
            if current_model is None:
                continue
            current_model = module_of(current_model)
            model_timestep = timestep_for(pipe, latents, t, None)
            pred = guided_velocity(
                current_model,
                args.student_adapter,
                latents,
                model_timestep,
                prompt_embeds,
                negative_prompt_embeds,
                args.sample_guidance_scale,
                image_embeds=None,
            )
            latents = pipe.scheduler.step(pred.to(latents.dtype), t, latents, return_dict=False)[0].float()
    return latents.detach().cpu()


@torch.no_grad()
def save_teacher_student_samples(
    pipe: WanImageToVideoPipeline,
    args,
    save_dir: str,
    global_step: int,
    prompt: str,
    prompt_id: int,
    teacher_latents: torch.Tensor,
    device: torch.device,
) -> None:
    sample_dir = Path(save_dir) / "samples" / f"step_{global_step:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed + int(prompt_id)

    teacher_path = sample_dir / f"prompt_{int(prompt_id):06d}_teacher_ti2v.mp4"
    student_path = sample_dir / f"prompt_{int(prompt_id):06d}_student_t2v.mp4"
    meta_path = sample_dir / f"prompt_{int(prompt_id):06d}.json"

    teacher_frames = decode_latents_to_frames(pipe, teacher_latents[:1].to(device), output_type="pil")[0]
    export_to_video(teacher_frames, str(teacher_path), fps=args.save_sample_fps)
    del teacher_frames

    student_latents = generate_t2v_student_latents(pipe, prompt, args, device, seed=seed)
    student_frames = decode_latents_to_frames(pipe, student_latents.to(device), output_type="pil")[0]
    export_to_video(student_frames, str(student_path), fps=args.save_sample_fps)
    del student_frames, student_latents

    meta = {
        "step": global_step,
        "prompt_id": int(prompt_id),
        "prompt": prompt,
        "seed": seed,
        "teacher_video": str(teacher_path),
        "student_video": str(student_path),
        "sample_inference_steps": args.sample_inference_steps or args.teacher_inference_steps,
        "sample_guidance_scale": args.sample_guidance_scale,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.getLogger(__name__).info(f"Saved teacher/student sample videos to {sample_dir}")


def make_scheduler(pipe: WanImageToVideoPipeline):
    return pipe.scheduler.__class__.from_config(pipe.scheduler.config)


def sample_traj_step_indices(
    timesteps: torch.Tensor,
    boundary_ratio: float | None,
    num_train_timesteps: int,
    high_count: int,
    low_count: int,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    ts = timesteps.detach().cpu().tolist()
    num_steps = len(ts)
    if boundary_ratio is not None:
        boundary_t = boundary_ratio * num_train_timesteps
        high_pool = [i for i, t in enumerate(ts) if t >= boundary_t]
        low_pool = [i for i, t in enumerate(ts) if t < boundary_t]
    else:
        mid = max(1, num_steps // 2)
        high_pool = list(range(0, mid))
        low_pool = list(range(mid, num_steps))

                                                                    
    if not high_pool or not low_pool:
        mid = max(1, num_steps // 2)
        high_pool = list(range(0, mid))
        low_pool = list(range(mid, num_steps))

    def _pick(pool: list[int], count: int) -> list[int]:
        count = max(1, min(count, len(pool)))
        perm = torch.randperm(len(pool), device=device)[:count].cpu().tolist()
        return sorted(pool[i] for i in perm)

    return _pick(high_pool, high_count), _pick(low_pool, low_count)


@torch.no_grad()
def run_teacher_trajectory_collect_velocity(
    pipe: WanImageToVideoPipeline,
    teacher_prompt: str,
    first_image,
    args,
    device: torch.device,
    seed: int,
    collect_indices: set[int],
    scheduler,
    num_steps: int,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    was_training = []
    for model in (pipe.transformer, pipe.transformer_2):
        if model is not None:
            target = module_of(model)
            was_training.append((target, target.training))
            target.eval()

    try:
        with only_active_adapter(pipe.transformer, args.teacher_adapter), \
             only_active_adapter(pipe.transformer_2, args.teacher_adapter):
            do_cfg = args.teacher_guidance_scale > 1.0
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt=[teacher_prompt],
                negative_prompt=[args.negative_prompt],
                do_classifier_free_guidance=do_cfg,
                num_videos_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
                device=device,
            )
            prompt_embeds = prompt_embeds.to(device=device)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device)

            scheduler.set_timesteps(num_steps, device=device)
            latents = initial_latents(pipe, 1, args.height, args.width, args.num_frames, device, seed=seed)
            condition, first_frame_mask, image_embeds = prepare_ti2v_condition(
                pipe, [first_image], latents, args.height, args.width, args.num_frames, device
            )

            traj_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for step_idx, t in enumerate(scheduler.timesteps):
                t = t.to(device=device, dtype=torch.float32)
                current_model, stage = choose_model(pipe, t)
                if current_model is None:
                    continue
                if pipe.config.expand_timesteps:
                    model_input = (1.0 - first_frame_mask) * condition + first_frame_mask * latents
                    model_timestep = timestep_for(pipe, latents, t, first_frame_mask)
                else:
                    model_input = torch.cat([latents, condition], dim=1)
                    model_timestep = timestep_for(pipe, latents, t, None)

                guidance = args.teacher_guidance_scale
                if stage == "low" and args.teacher_guidance_scale_2 is not None:
                    guidance = args.teacher_guidance_scale_2
                pred = guided_velocity(
                    current_model, args.teacher_adapter, model_input, model_timestep,
                    prompt_embeds, negative_prompt_embeds, guidance, image_embeds,
                )

                if step_idx in collect_indices:
                                                                                           
                    traj_states[step_idx] = (latents.detach().clone(), pred.detach().clone())

                latents = scheduler.step(pred.to(latents.dtype), t, latents, return_dict=False)[0].float()

            if pipe.config.expand_timesteps:
                latents = (1.0 - first_frame_mask) * condition + first_frame_mask * latents
            return traj_states, latents.detach(), condition.detach(), first_frame_mask.detach(), image_embeds
    finally:
        for target, train_state in was_training:
            target.train(train_state)


def hpsd_loss(
    pipe: WanImageToVideoPipeline,
    first_frame_manager: FirstFrameManager,
    prompts: list[str],
    ids: list[int],
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:

    teacher_scheduler = make_scheduler(pipe)

    num_steps = args.inference_steps or args.teacher_inference_steps
    teacher_scheduler.set_timesteps(num_steps, device=device)
    timesteps = teacher_scheduler.timesteps

    if args.anchor_indices is not None:
                                                                     
        collect_indices = sorted({i for i in args.anchor_indices if 0 <= i < len(timesteps)})
    else:
        boundary_ratio = pipe.config.boundary_ratio if pipe.transformer_2 is not None else None
        num_train_timesteps = pipe.scheduler.config.num_train_timesteps
        high_indices, low_indices = sample_traj_step_indices(
            timesteps, boundary_ratio, num_train_timesteps, args.traj_high_steps, args.traj_low_steps, device
        )
        collect_indices = sorted(set(high_indices + low_indices))
    collect_set = set(collect_indices)

                                                                      
                                                     
    hybrid_K_default = int(getattr(args, "hybrid_continue_steps", 0) or 0)
    if args.sub_traj_length_list is not None:
        k_list = list(args.sub_traj_length_list)
        if k_list:
            pad = k_list[-1]
        else:
            pad = 0
        while len(k_list) < len(collect_indices):
            k_list.append(pad)
        continue_steps_map = dict(zip(collect_indices, k_list[:len(collect_indices)]))
    else:
        continue_steps_map = {idx: hybrid_K_default for idx in collect_indices}

    any_hybrid = any(k > 0 for k in continue_steps_map.values())

    losses: list[torch.Tensor] = []
    teacher_final_latents: torch.Tensor | None = None

    for i, (prompt_id, prompt) in enumerate(zip(ids, prompts)):
        seed = args.seed + int(prompt_id)
        first_image, first_meta = first_frame_manager.ensure(prompt, prompt_id)
        teacher_prompt = first_meta.get("teacher_video_prompt_enhanced") or first_frame_manager.enhance_video_prompt(prompt, prompt_id)

        teacher_traj, teacher_final, condition, first_frame_mask, image_embeds = run_teacher_trajectory_collect_velocity(
            pipe, teacher_prompt, first_image, args, device, seed, collect_set, teacher_scheduler, num_steps
        )
        if i == 0:
            teacher_final_latents = teacher_final.detach().cpu()

        pe = prompt_embeds[i : i + 1]
        npe = negative_prompt_embeds[i : i + 1] if negative_prompt_embeds is not None else None
                                  
        non_ff_mask = first_frame_mask.to(device=device, dtype=torch.float32)
                                                                   
        teacher_pe = None
        teacher_npe = None
        if any_hybrid:
            do_cfg_t = args.teacher_guidance_scale > 1.0
            teacher_pe, teacher_npe = pipe.encode_prompt(
                prompt=[teacher_prompt],
                negative_prompt=[args.negative_prompt],
                do_classifier_free_guidance=do_cfg_t,
                num_videos_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
                device=device,
            )
            teacher_pe = teacher_pe.to(device=device)
            if teacher_npe is not None:
                teacher_npe = teacher_npe.to(device=device)

        for idx in collect_indices:
            if idx not in teacher_traj:
                continue
            xt_teacher, v_teacher_collected = teacher_traj[idx]
            t = timesteps[idx].to(device=device, dtype=torch.float32)
                                               
            sigma = teacher_scheduler.sigmas[idx].to(device=device, dtype=torch.float32)
                                                                                      
            ff_noise = torch.randn_like(condition)
            ff_noisy = sigma * ff_noise + (1.0 - sigma) * condition
                                                               
            xt_start = (1.0 - non_ff_mask) * ff_noisy + non_ff_mask * xt_teacher.to(device=device, dtype=torch.float32)

            if continue_steps_map[idx] > 0:
                                                                                                                                       
                effective_K = max(0, min(continue_steps_map[idx], num_steps - 1 - idx))

                xt_hybrid = xt_start.detach()
                if effective_K > 0:
                                                                                                                          
                    with torch.no_grad(), eval_mode(pipe.transformer, pipe.transformer_2):
                        for k in range(effective_K):
                            s_idx = idx + k
                            t_k = timesteps[s_idx].to(device=device, dtype=torch.float32)
                            cur_model_k, _ = choose_model(pipe, t_k)
                            if cur_model_k is None:
                                continue
                            ts_k = timestep_for(pipe, xt_hybrid, t_k, None)
                            v_k = guided_velocity(
                                cur_model_k, args.student_adapter, xt_hybrid, ts_k,
                                pe, npe, args.train_guidance_scale, image_embeds=None,
                            )
                            dt_k = (teacher_scheduler.sigmas[s_idx + 1] - teacher_scheduler.sigmas[s_idx]).to(
                                device=device, dtype=torch.float32
                            )
                            xt_hybrid = xt_hybrid + dt_k * v_k
                    xt_hybrid = xt_hybrid.detach()

                new_idx = idx + effective_K
                t_new = timesteps[new_idx].to(device=device, dtype=torch.float32)
                                                      
                fake_student_latent = (1.0 - non_ff_mask) * condition + non_ff_mask * xt_hybrid
                with torch.no_grad(), eval_mode(pipe.transformer, pipe.transformer_2):
                    cur_model_t, stage = choose_model(pipe, t_new)
                    if cur_model_t is not None:
                        teacher_ts = timestep_for(pipe, fake_student_latent, t_new, first_frame_mask)
                        guidance_t = args.teacher_guidance_scale
                        if stage == "low" and args.teacher_guidance_scale_2 is not None:
                            guidance_t = args.teacher_guidance_scale_2
                        v_teacher = guided_velocity(
                            cur_model_t, args.teacher_adapter, fake_student_latent, teacher_ts,
                            teacher_pe, teacher_npe, guidance_t, image_embeds,
                        )
                    else:
                        v_teacher = v_teacher_collected.float().to(device=device)

                                                                              
                cur_model_s, _ = choose_model(pipe, t_new)
                if cur_model_s is None:
                    continue
                student_ts = timestep_for(pipe, xt_hybrid, t_new, None)
                v_student = guided_velocity(
                    cur_model_s, args.student_adapter, xt_hybrid, student_ts,
                    pe, npe, args.train_guidance_scale, image_embeds=None,
                )
            else:
                                                                                     
                current_model, _ = choose_model(pipe, t)
                if current_model is None:
                    continue
                                                                                
                model_timestep = timestep_for(pipe, xt_start, t, None)
                v_student = guided_velocity(
                    current_model, args.student_adapter, xt_start, model_timestep,
                    pe, npe, args.train_guidance_scale, image_embeds=None,
                )
                v_teacher = v_teacher_collected

                                                        
            v_s = v_student.float() * non_ff_mask
            v_t = v_teacher.float().to(device=device) * non_ff_mask
            losses.append(F.mse_loss(v_s, v_t.detach()))

    if not losses:
        raise RuntimeError("No HPSD loss was computed; check WAN transformer configuration.")
    return sum(losses) / len(losses), [x.detach() for x in losses], teacher_final_latents


def precompute_first_frames(args, dataset: VideoPromptDataset, accelerator: Accelerator) -> None:
    if not args.precompute_first_frames:
        return
    if accelerator.is_main_process:
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
            for i in tqdm(range(len(dataset)), desc="first-frame cache"):
                item = dataset[i]
                manager.ensure(item["prompt"], item["id"])
        finally:
            manager.close()
    accelerator.wait_for_everyone()


def main(args) -> None:
    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config) if args.deepspeed_config else None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        deepspeed_plugin=deepspeed_plugin,
    )
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    save_dir = os.path.join(args.output_dir, args.exp_name)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        local_logger = make_logger(save_dir)
        local_logger.info(f"Experiment directory: {save_dir}")

    dataset = VideoPromptDataset(
        args.prompt_file,
        prompt_key=args.prompt_key,
        max_samples=args.max_train_samples,
        shuffle=args.shuffle_prompts,
        shuffle_seed=args.prompt_shuffle_seed,
    )
    precompute_first_frames(args, dataset, accelerator)

    device = accelerator.device
    transformer_dtype = dtype_from_name(args.transformer_dtype)
    vae_dtype = dtype_from_name(args.vae_dtype)

    vae = AutoencoderKLWan.from_pretrained(args.wan_model_id, subfolder="vae", torch_dtype=vae_dtype)
    pipe = WanImageToVideoPipeline.from_pretrained(args.wan_model_id, vae=vae, torch_dtype=transformer_dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if pipe.image_encoder is not None:
        pipe.image_encoder.requires_grad_(False)
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    first_frame_manager = FirstFrameManager(
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

    targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    pipe.transformer = init_dual_lora_model(
        pipe.transformer,
        args.lora_rank,
        args.lora_alpha,
        targets,
        student_adapter=args.student_adapter,
        teacher_adapter=args.teacher_adapter,
        teacher_init_from_student=True,
    )
    if pipe.transformer_2 is not None:
        pipe.transformer_2 = init_dual_lora_model(
            pipe.transformer_2,
            args.lora_rank,
            args.lora_alpha,
            targets,
            student_adapter=args.student_adapter,
            teacher_adapter=args.teacher_adapter,
            teacher_init_from_student=True,
        )

    if args.enable_gc:
        pipe.transformer.enable_gradient_checkpointing()
        if pipe.transformer_2 is not None:
            pipe.transformer_2.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        trainable_parameters(pipe.transformer, pipe.transformer_2),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_prompts,
        drop_last=True,
    )

    if pipe.transformer_2 is not None:
        pipe.transformer, pipe.transformer_2, optimizer, dataloader = accelerator.prepare(
            pipe.transformer, pipe.transformer_2, optimizer, dataloader
        )
    else:
        pipe.transformer, optimizer, dataloader = accelerator.prepare(pipe.transformer, optimizer, dataloader)

    log_path = os.path.join(save_dir, "loss_log.jsonl")
    if accelerator.is_main_process:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="HPSD steps")

    try:
        for epoch in range(args.epochs):
            for batch in dataloader:
                accumulate_models = (pipe.transformer, pipe.transformer_2) if pipe.transformer_2 is not None else (pipe.transformer,)
                with accelerator.accumulate(*accumulate_models):
                    prompts = batch["prompts"]
                    ids = batch["ids"]
                    bsz = len(prompts)

                    with torch.no_grad():
                        do_cfg = args.train_guidance_scale > 1.0
                        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                            prompt=prompts,
                            negative_prompt=[args.negative_prompt] * bsz,
                            do_classifier_free_guidance=do_cfg,
                            num_videos_per_prompt=1,
                            max_sequence_length=args.max_sequence_length,
                            device=device,
                        )
                        prompt_embeds = prompt_embeds.to(device=device)
                        if negative_prompt_embeds is not None:
                            negative_prompt_embeds = negative_prompt_embeds.to(device=device)

                    loss, per_t_losses, teacher_final_latents = hpsd_loss(
                        pipe, first_frame_manager, prompts, ids, prompt_embeds, negative_prompt_embeds, args, device
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(trainable_parameters(pipe.transformer, pipe.transformer_2), args.max_grad_norm)
                    else:
                        grad_norm = None
                    optimizer.step()
                    if accelerator.sync_gradients:
                        ema_update_lora_adapter(
                            pipe.transformer,
                            src_adapter=args.student_adapter,
                            dst_adapter=args.teacher_adapter,
                            ema_decay=args.ema_decay,
                        )
                        if pipe.transformer_2 is not None:
                            ema_update_lora_adapter(
                                pipe.transformer_2,
                                src_adapter=args.student_adapter,
                                dst_adapter=args.teacher_adapter,
                                ema_decay=args.ema_decay,
                            )
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)
                    reduced_loss = accelerator.gather(loss.detach()).mean().item()
                    logs = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": reduced_loss,
                        "loss_per_sampled_timestep_local": [x.detach().cpu().item() for x in per_t_losses],
                        "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                    }
                    progress_bar.set_postfix(loss=logs["loss"])
                    if accelerator.is_main_process and global_step % args.log_steps == 0:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(logs, ensure_ascii=False) + "\n")

                    if args.save_sample_every_steps > 0 and global_step % args.save_sample_every_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            save_teacher_student_samples(
                                pipe,
                                args,
                                save_dir,
                                global_step,
                                prompts[0],
                                int(ids[0]),
                                teacher_final_latents[:1],
                                device,
                            )
                            gc.collect()
                            torch.cuda.empty_cache()
                        accelerator.wait_for_everyone()

                    if global_step % args.checkpoint_steps == 0 or global_step >= args.max_train_steps:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            step_dir = os.path.join(checkpoint_dir, f"step_{global_step}")
                            os.makedirs(step_dir, exist_ok=True)

                            def _save_student_adapter(model, subdir):
                                save_path = os.path.join(step_dir, subdir)
                                unwrap_model(model, accelerator).save_pretrained(
                                    save_path,
                                    selected_adapters=[args.student_adapter],
                                )
                                                               
                                readme = os.path.join(save_path, "README.md")
                                if os.path.exists(readme):
                                    os.remove(readme)

                            _save_student_adapter(pipe.transformer, "transformer")
                            if pipe.transformer_2 is not None:
                                _save_student_adapter(pipe.transformer_2, "transformer_2")

                    if global_step >= args.max_train_steps:
                        break
            if global_step >= args.max_train_steps:
                break
    finally:
        first_frame_manager.close()

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
