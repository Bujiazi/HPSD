from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image
                       
try:
    from flash_attn import flash_attn_varlen_func              

    FLASH_VER = 2
except ModuleNotFoundError:
    FLASH_VER = None


DEFAULT_PROMPT_TEMPLATE = """Act as an expert AI Visual Director and Prompt Engineer specializing in Image-to-Video (I2V) and Text-to-Image (T2I) workflows. Your task is to design a highly detailed Text-to-Image prompt that will serve as the first frame for a video generation model. You will be provided with a complete Video Prompt.

Follow these rules:
1. Convert dynamic actions into a static, ready-to-move opening frame.
2. Preserve the subject, environment, style, camera angle, lighting, and atmosphere.
3. Output only one English image prompt, without explanations.

Examples:
-[Input]: A young woman walks along a rainy city street at night, turning to look at a neon sign, then continues walking as her umbrella spins.
[Output]: A young woman standing mid-stride on a rain-soaked city street at night, holding an open umbrella, her body slightly turned toward a glowing pink-and-blue neon sign; wet asphalt reflecting the neon glow, soft drizzle catching the streetlight, shallow depth of field, cinematic moody atmosphere, 35mm lens, eye-level angle.

-[Input]: A golden retriever runs across a sunny beach, splashing through shallow water, shakes off water, then lies down in the sand.
[Output]: A golden retriever poised at the water's edge on a sunlit beach, front paws just touching the shallow waves, alert and ready to spring; bright midday sun, sparkling turquoise water, soft white sand, vibrant natural lighting, wide-angle low shot, photorealistic.

-[Input]: Steam rises from a cup of coffee on a wooden table as morning sunlight streams through a window, curtains gently swaying.
[Output]: A ceramic cup of hot coffee on a rustic wooden table, a faint wisp of steam just beginning to curl upward, soft golden morning light streaming through a nearby window casting long warm streaks, sheer curtains slightly lifted, cozy quiet atmosphere, still-life composition, 50mm lens, warm tones.

-[Input]: An ancient-style swordsman stands in a bamboo grove, draws his sword slowly, then turns to face the wind as leaves swirl around him.
[Output]: An ancient swordsman standing poised in a misty bamboo grove, hand resting on the hilt of an undrawn sword at his side, calm determined expression; tall green bamboo framing the shot, soft diffused fog, shafts of pale light filtering through the canopy, three-quarter front view, balanced composition.

Now, convert the given Video Prompt.
[Input Video Prompt] : {video_prompt}
""".strip()


LM_ZH_SYS_PROMPT = """你是一位专注TI2V（给定首帧图像的文本到视频）任务的Prompt优化师。你的任务是将给定的视频Prompt改写为适配已有高质量首帧、以运动为导向的TI2V Prompt，在不改变原意的前提下提升“场景内运动可执行性”。你将会收到一条完整的视频Prompt。

改写原则：
1) 忠实原意：严格保留主体、场景、风格与关键物体，不新增冲突设定。
2) 运动优先：重点描写已有元素的运动，如人物肢体/表情微动作、衣物摆动、光影变化、烟雾/水流/树叶等环境动态。
3) 时序清晰：可用“先/随后/最后”等连接词组织动作链，但保持单场景连续性。
4) 适度细节：避免堆砌空泛修辞和重复画质词，突出可执行动作信息。
5) 语言纯净：只输出最终改写后的Prompt正文，不要解释、不要分点、不要前后缀。
6) 严禁输出任何思考过程、分析步骤、解释文本（例如“Here is my thinking process”）。
7) 长度建议：中文约80-130字，优先保证运动稳定性与生成一致性。
"""


LM_EN_SYS_PROMPT = """You are a prompt engineer specialized for TI2V (text-to-video with a given first frame). Your task is to rewrite a given Video Prompt into a motion-optimized TI2V prompt that works with an already provided high-quality first frame. Your core goal is to improve in-scene motion executability while preserving intent. You will be provided with a complete Video Prompt.

Rewrite rules:
1) Preserve intent: keep subject, scene, style, and key objects; do not add conflicting settings.
2) Motion-first: focus on motion of existing elements, such as body micro-actions, facial changes, fabric movement, light/shadow variation, smoke/water/foliage dynamics.
3) Clear timeline: use concise temporal connectors (first, then, finally) while keeping single-scene continuity.
4) Moderate detail: avoid verbose quality stacking; prioritize actionable motion cues.
5) Output only the final rewritten prompt text with no explanation or list formatting.
6) Never output any reasoning process, analysis steps, or meta explanation.
7) Preferred length: around 80-130 words, prioritizing motion stability and temporal consistency.
"""


FEW_SHOTS_ZH = [
    {
        "user": "一个女孩站在海边",
        "assistant": "海边黄昏里，一个女孩面向海风站立，发丝持续轻摆，裙摆在风里有节奏地起伏。她先微微抬眼看向远处海面，随后缓慢转头，呼吸与肩颈动作自然连贯；背景中浪花反复拍岸，近处水雾时聚时散，地面倒影随波纹轻微晃动。整体保持同一场景、稳定构图，仅呈现场内元素的连续运动。",
    },
    {
        "user": "森林里一只鹿",
        "assistant": "清晨薄雾森林中，一只鹿先短暂停顿，耳朵轻颤后缓慢迈步，鼻尖轻嗅空气并不时抬头观察。它的步伐由慢到稳，颈部和背部肌肉随呼吸细微起伏；周围树叶被微风带动，草尖轻晃，薄雾在林间缓慢流动。全程维持单场景连续表达，重点突出鹿与环境的自然动态。",
    },
    {
        "user": "古风男子在庭院里喝茶",
        "assistant": "古风庭院内，男子端坐案前，先抬手执杯，袖口随动作自然垂落，茶面泛起细小涟漪；随后他轻抿一口，放杯时指尖轻触杯沿，目光缓慢上移。背景竹影随风轻晃，檐下风铃微微摆动，桌旁薄烟缓慢上升并轻轻散开。保持同一场景与稳定画面，以人物与环境细节运动推进时序。",
    },
]


FEW_SHOTS_EN = [
    {
        "user": "A woman standing by the sea",
        "assistant": "At seaside dusk, a woman stands in the wind as her hair sways continuously and the hem of her dress rises and settles in a gentle rhythm. She first lifts her gaze toward the horizon, then slowly turns her head, with natural breathing and subtle shoulder motion. Waves repeatedly break behind her, sea spray drifts in and out, and reflections on wet ground shimmer slightly. Keep one continuous scene and stable framing, focusing on in-scene motion only.",
    },
    {
        "user": "A deer in the forest",
        "assistant": "In a misty morning forest, a deer pauses briefly, ears twitching, then begins to step forward at a calm pace. It sniffs the air, occasionally raises its head, and shows subtle neck and back movement with each breath. Nearby leaves sway lightly, grass tips tremble, and thin fog moves slowly between trees. Keep the scene continuous and stable, emphasizing natural motion of the deer and surrounding elements rather than dramatic camera movement.",
    },
    {
        "user": "An ancient-style man drinking tea in a courtyard",
        "assistant": "In a traditional courtyard, a man sits at a tea table, lifts the cup, and his sleeve falls naturally as small ripples spread across the tea surface. He takes a sip, then sets the cup down with a light fingertip touch on the rim, raising his eyes slowly afterward. Bamboo shadows sway softly, a hanging chime moves slightly, and a thin trail of steam rises and disperses near the table. Keep one scene and stable composition, driven by character and environment motion.",
    },
]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def safe_one_line(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split()).strip()


def qwen_single_process_device_map():
    import torch

    if torch.cuda.is_available():
        return {"": torch.cuda.current_device()}
    return "cpu"


class LocalQwenPromptGenerator:
    def __init__(self, model_path: str, torch_dtype: str = "bfloat16", attn_implementation: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": dtype_map[torch_dtype],
            "device_map": qwen_single_process_device_map(),
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(**kwargs)
        self.model.eval()

    def __call__(self, query: str, max_new_tokens: int = 512) -> str:
        import torch

        messages = [{"role": "user", "content": query}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        return safe_one_line(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()


class LocalQwenTI2VPromptExpander:
    def __init__(self, model_path: str, tokenizer=None, model=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self._owns_model = model is None or tokenizer is None

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if FLASH_VER == 2 else torch.float16,
                attn_implementation="flash_attention_2" if FLASH_VER == 2 else None,
                device_map=qwen_single_process_device_map(),
            )
            self.model.eval()

    def _build_messages(self, prompt: str, tar_lang: str):
        if tar_lang == "zh":
            system_prompt = LM_ZH_SYS_PROMPT
            few_shots = FEW_SHOTS_ZH
            final_user_prompt = f"{prompt}\n\n请仅输出最终改写后的Prompt正文，不要输出思考过程、分析、标题或前后缀。"
        else:
            system_prompt = LM_EN_SYS_PROMPT
            few_shots = FEW_SHOTS_EN
            final_user_prompt = (
                f"{prompt}\n\nOutput only the final rewritten prompt text. "
                "No thinking process, no analysis, no title, no prefix."
            )

        messages = [{"role": "system", "content": system_prompt}]
        for shot in few_shots:
            messages.append({"role": "user", "content": shot["user"]})
            messages.append({"role": "assistant", "content": shot["assistant"]})

        messages.append({"role": "user", "content": final_user_prompt})
        return messages

    @staticmethod
    def _looks_like_reasoning(text: str) -> bool:
        lowered = text.lower()
        hints = [
            "here's a thinking process",
            "thinking process",
            "analyze user input",
            "apply rewrite rules",
            "draft construction",
            "reasoning",
            "step-by-step",
        ]
        return any(h in lowered for h in hints)

    @staticmethod
    def _postprocess_output(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

        marker_patterns = [
            r"(?i)final\s*prompt\s*:\s*",
            r"(?i)rewritten\s*prompt\s*:\s*",
            r"(?i)video\s*prompt\s*:\s*",
        ]
        for pattern in marker_patterns:
            matched = re.search(pattern, cleaned)
            if matched:
                cleaned = cleaned[matched.end() :].strip()

        if "\n\n" in cleaned and LocalQwenTI2VPromptExpander._looks_like_reasoning(cleaned):
            cleaned = cleaned.split("\n\n")[-1].strip()

        return cleaned

    def _apply_chat_template_no_think(self, messages):
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def enhance(
        self,
        prompt: str,
        tar_lang: str = "en",
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        import torch

        messages = self._build_messages(prompt, tar_lang)
        text = self._apply_chat_template_no_think(messages)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        expanded_prompt = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        expanded_prompt = self._postprocess_output(expanded_prompt)

        if self._looks_like_reasoning(expanded_prompt):
            retry_messages = self._build_messages(prompt, tar_lang)
            retry_messages.append(
                {
                    "role": "user",
                    "content": "再次强调：只输出最终Prompt文本本身。禁止输出任何思考或分析。",
                }
            )
            retry_text = self._apply_chat_template_no_think(retry_messages)
            retry_inputs = self.tokenizer([retry_text], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                retry_ids = self.model.generate(
                    **retry_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            retry_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(retry_inputs.input_ids, retry_ids)
            ]
            retry_prompt = self.tokenizer.batch_decode(retry_ids, skip_special_tokens=True)[0].strip()
            retry_prompt = self._postprocess_output(retry_prompt)
            if retry_prompt:
                expanded_prompt = retry_prompt

        return expanded_prompt

    def close(self) -> None:
        if self._owns_model:
            del self.model
            del self.tokenizer
        gc.collect()


class ZImageFirstFrameGenerator:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from diffusers import ZImagePipeline

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            dtype = "float32"
        self.device = device
        self.pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=dtype_map[dtype], low_cpu_mem_usage=False)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(
        self,
        prompt: str,
        height: int,
        width: int,
        seed: int,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        import torch

        generator_device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        kwargs: dict[str, Any] = {}
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=generator_device).manual_seed(seed),
            **kwargs,
        ).images[0]
        return image.convert("RGB")

    def close(self) -> None:
        del self.pipe
        gc.collect()


class FirstFrameManager:
    def __init__(
        self,
        cache_dir: str,
        qwen_model_path: str,
        zimage_model_path: str,
        template_path: str | None = None,
        placeholder: str = "{请在这里填入你的视频Prompt}",
        height: int = 704,
        width: int = 1280,
        zimage_steps: int = 9,
        zimage_guidance_scale: float = 0.0,
        base_seed: int = 42,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.image_dir = self.cache_dir / "images"
        self.meta_dir = self.cache_dir / "metadata"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.qwen_model_path = qwen_model_path
        self.zimage_model_path = zimage_model_path
        self.placeholder = placeholder
        self.height = height
        self.width = width
        self.zimage_steps = zimage_steps
        self.zimage_guidance_scale = zimage_guidance_scale
        self.base_seed = base_seed
        self._llm: LocalQwenPromptGenerator | None = None
        self._prompt_expander: LocalQwenTI2VPromptExpander | None = None
        self._zimage: ZImageFirstFrameGenerator | None = None

        if template_path and Path(template_path).exists():
            self.template = Path(template_path).read_text(encoding="utf-8")
        else:
            self.template = DEFAULT_PROMPT_TEMPLATE

    def paths_for(self, prompt: str, prompt_id: int | None = None) -> tuple[Path, Path]:
        h = prompt_hash(prompt)
        prefix = f"{prompt_id:06d}_" if prompt_id is not None else ""
        stem = f"{prefix}{h[:16]}"
        return self.image_dir / f"{stem}.png", self.meta_dir / f"{stem}.json"

    def build_llm_query(self, video_prompt: str) -> str:
        if self.placeholder in self.template:
            return self.template.replace(self.placeholder, video_prompt)
        if "{video_prompt}" in self.template:
            return self.template.replace("{video_prompt}", video_prompt)
        return f"{self.template}\n\n[Input Video Prompt]: {video_prompt}\n"

    @staticmethod
    def detect_prompt_language(video_prompt: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", video_prompt) else "en"

    def _generate_enhanced(self, video_prompt: str) -> str:
        if self._prompt_expander is None:
            if self._llm is not None:
                self._prompt_expander = LocalQwenTI2VPromptExpander(
                    self.qwen_model_path,
                    tokenizer=self._llm.tokenizer,
                    model=self._llm.model,
                )
            else:
                self._prompt_expander = LocalQwenTI2VPromptExpander(self.qwen_model_path)
        tar_lang = self.detect_prompt_language(video_prompt)
        enhanced = self._prompt_expander.enhance(video_prompt, tar_lang=tar_lang)
        enhanced = safe_one_line(enhanced)
        return enhanced if enhanced else video_prompt

    def _load_enhanced_prompt(self, prompt: str, prompt_id: int) -> str | None:
        _, meta_path = self.paths_for(prompt, prompt_id)
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        cached = meta.get("teacher_video_prompt_enhanced")
        return safe_one_line(cached) if cached else None

    def _save_enhanced_prompt(self, prompt: str, prompt_id: int, enhanced: str) -> None:
        _, meta_path = self.paths_for(prompt, prompt_id)
        if not meta_path.exists():
            return
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        meta["teacher_video_prompt_enhanced"] = enhanced
        tmp_meta = meta_path.with_suffix(f".tmp.{os.getpid()}.json")
        with tmp_meta.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp_meta, meta_path)

    def enhance_video_prompt(self, video_prompt: str, prompt_id: int | None = None) -> str:
        if prompt_id is not None:
            cached = self._load_enhanced_prompt(video_prompt, prompt_id)
            if cached is not None:
                return cached
        enhanced = self._generate_enhanced(video_prompt)
        if prompt_id is not None:
            self._save_enhanced_prompt(video_prompt, prompt_id, enhanced)
        return enhanced

    def load_cached(self, prompt: str, prompt_id: int | None = None) -> tuple[Image.Image, dict[str, Any]]:
        image_path, meta_path = self.paths_for(prompt, prompt_id)
        if not image_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"First-frame cache missing for prompt_id={prompt_id}: {image_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return Image.open(image_path).convert("RGB"), meta

    def ensure(self, prompt: str, prompt_id: int | None = None) -> tuple[Image.Image, dict[str, Any]]:

        image_path, meta_path = self.paths_for(prompt, prompt_id)
        if image_path.exists() and meta_path.exists():
            return self.load_cached(prompt, prompt_id)

        if self._llm is None:
            self._llm = LocalQwenPromptGenerator(self.qwen_model_path)
        if self._zimage is None:
            self._zimage = ZImageFirstFrameGenerator(self.zimage_model_path)

        first_frame_prompt = self._llm(self.build_llm_query(prompt))
        seed = self.base_seed + (int(prompt_id) if prompt_id is not None else int(prompt_hash(prompt)[:8], 16) % 100000)
        image = self._zimage(
            first_frame_prompt,
            height=self.height,
            width=self.width,
            seed=seed,
            num_inference_steps=self.zimage_steps,
            guidance_scale=self.zimage_guidance_scale,
        )

        teacher_video_prompt_enhanced = self._generate_enhanced(prompt)

        tmp_image = image_path.with_suffix(f".tmp.{os.getpid()}.png")
        tmp_meta = meta_path.with_suffix(f".tmp.{os.getpid()}.json")
        image.save(tmp_image)
        meta = {
            "prompt_id": prompt_id,
            "video_prompt": prompt,
            "first_frame_prompt": first_frame_prompt,
            "teacher_video_prompt_enhanced": teacher_video_prompt_enhanced,
            "image_path": str(image_path),
            "height": self.height,
            "width": self.width,
            "seed": seed,
            "zimage_steps": self.zimage_steps,
            "zimage_guidance_scale": self.zimage_guidance_scale,
        }
        with tmp_meta.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp_image, image_path)
        os.replace(tmp_meta, meta_path)
        return image, meta

    def close(self) -> None:
        if self._llm is not None:
            self._llm.close()
            self._llm = None
        if self._prompt_expander is not None:
            self._prompt_expander.close()
            self._prompt_expander = None
        if self._zimage is not None:
            self._zimage.close()
            self._zimage = None
        gc.collect()
