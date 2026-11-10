"""Model config read from a Hugging Face checkpoint directory."""

import json
import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_eps: float
    rope_theta: float
    tie_embeddings: bool
    qkv_bias: bool
    max_position: int
    eos_token_ids: tuple

    @classmethod
    def from_dict(cls, d, generation=None):
        mt = d.get("model_type")
        if mt not in ("llama", "qwen2", "mistral"):
            raise ValueError(f"unsupported model_type {mt!r}")
        rs = d.get("rope_scaling")
        if rs and rs.get("rope_type", rs.get("type")) not in (None, "default"):
            raise ValueError(f"unsupported rope_scaling {rs}")
        if d.get("use_sliding_window"):
            raise ValueError("sliding window attention is not supported")
        heads = d["num_attention_heads"]
        eos = set()
        for src in (d, generation or {}):
            e = src.get("eos_token_id")
            if isinstance(e, int):
                eos.add(e)
            elif e:
                eos.update(e)
        # rope_theta moved into rope_parameters in newer configs
        theta = d.get("rope_theta") or (d.get("rope_parameters") or {}).get("rope_theta", 10000.0)
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_hidden_layers"],
            num_heads=heads,
            num_kv_heads=d.get("num_key_value_heads") or heads,
            head_dim=d.get("head_dim") or d["hidden_size"] // heads,
            rms_eps=d["rms_norm_eps"],
            rope_theta=float(theta),
            tie_embeddings=bool(d.get("tie_word_embeddings", False)),
            qkv_bias=(mt == "qwen2") or bool(d.get("attention_bias", False)),
            max_position=d.get("max_position_embeddings", 4096),
            eos_token_ids=tuple(sorted(eos)),
        )

    @classmethod
    def from_dir(cls, path):
        with open(os.path.join(path, "config.json")) as f:
            d = json.load(f)
        gen = None
        gp = os.path.join(path, "generation_config.json")
        if os.path.exists(gp):
            with open(gp) as f:
                gen = json.load(f)
        return cls.from_dict(d, gen)
