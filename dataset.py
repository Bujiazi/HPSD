import json
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class VideoPromptDataset(Dataset):
    def __init__(
        self,
        path: str,
        prompt_key: str = "prompt",
        max_samples: int | None = None,
        start: int = 0,
        shuffle: bool = False,
        shuffle_seed: int = 42,
    ):
        self.path = Path(path).expanduser().resolve()
        self.prompt_key = prompt_key
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        if not self.path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.path}")

        if self.path.suffix.lower() == ".jsonl":
            self.rows = self._load_jsonl(max_samples, start)
        else:
            self.rows = self._load_txt(max_samples, start)
        if not self.rows:
            raise ValueError(f"No valid prompts found in {self.path} (start={start}, max_samples={max_samples})")

    def _slice(self, all_rows: list[dict[str, Any]], max_samples: int | None, start: int) -> list[dict[str, Any]]:

        if self.shuffle:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(all_rows)
        end = None if max_samples is None else start + max_samples
        return all_rows[start:end]

    def _load_txt(self, max_samples: int | None, start: int) -> list[dict[str, Any]]:
        all_rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                prompt = line.strip()
                if not prompt:
                    continue
                all_rows.append({"id": idx, "prompt": prompt})
        return self._slice(all_rows, max_samples, start)

    def _load_jsonl(self, max_samples: int | None, start: int) -> list[dict[str, Any]]:
        all_rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = str(item.get(self.prompt_key) or item.get("prompt") or item.get("caption") or "").strip()
                if not prompt:
                    continue
                all_rows.append({"id": item.get("id", idx), "prompt": prompt, "raw": item})
        return self._slice(all_rows, max_samples, start)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        raw_id = row["id"]
        return {"id": int(raw_id) if str(raw_id).isdigit() else index, "prompt": row["prompt"]}


def collate_prompts(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ids": [int(x["id"]) for x in examples],
        "prompts": [str(x["prompt"]) for x in examples],
    }
