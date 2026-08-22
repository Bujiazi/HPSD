from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model


def module_of(model):
    return getattr(model, "module", model)


def set_adapter_trainable(model, adapter_name: str, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if adapter_name in name:
            param.requires_grad = trainable


def set_active_adapter(model, adapter_name: str) -> None:
    target = module_of(model)
    if hasattr(target, "set_adapter"):
        target.set_adapter(adapter_name)


@torch.no_grad()
def copy_lora_adapter_weights(model, src_adapter: str, dst_adapter: str) -> None:
    target = module_of(model)
    named_params = dict(target.named_parameters())
    for name, src_param in named_params.items():
        if src_adapter not in name:
            continue
        dst_name = name.replace(src_adapter, dst_adapter)
        if dst_name in named_params:
            named_params[dst_name].data.copy_(src_param.data)


def init_dual_lora_model(
    model,
    lora_rank: int,
    lora_alpha: int,
    target_modules: list[str],
    student_adapter: str = "student",
    teacher_adapter: str = "teacher",
    teacher_init_from_student: bool = True,
):
    for param in model.parameters():
        param.requires_grad = False

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, config, adapter_name=student_adapter)
    model.add_adapter(teacher_adapter, config)
    model.set_adapter(student_adapter)

    set_adapter_trainable(model, student_adapter, True)
    set_adapter_trainable(model, teacher_adapter, False)

    if teacher_init_from_student:
        copy_lora_adapter_weights(model, src_adapter=student_adapter, dst_adapter=teacher_adapter)

    return model


@torch.no_grad()
def ema_update_lora_adapter(model, src_adapter: str, dst_adapter: str, ema_decay: float = 0.999) -> None:
    target = module_of(model)
    named_params = dict(target.named_parameters())

    for name, src_param in named_params.items():
        if src_adapter not in name:
            continue
        dst_name = name.replace(src_adapter, dst_adapter)
        if dst_name not in named_params:
            continue
        dst_param = named_params[dst_name]
        dst_param.data.mul_(ema_decay).add_(src_param.data, alpha=1.0 - ema_decay)


def trainable_parameters(*models) -> list[torch.nn.Parameter]:
    params = []
    for model in models:
        if model is None:
            continue
        params.extend([p for p in model.parameters() if p.requires_grad])
    return params
