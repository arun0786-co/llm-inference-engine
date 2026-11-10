"""Llama / Qwen2 decoder written directly against the checkpoint tensors.

The forward pass takes one flat batch of tokens from many sequences
("varlen"): sequence A's new tokens, then B's, and so on. Each sequence
can bring any number of new tokens (a prefill chunk, one decode token, or
the k+1 tokens of a speculative verify). Their keys and values are written
into the paged KV cache, and attention reads each sequence's context back
through its block table.
"""

import glob
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .config import ModelConfig


@dataclass
class AttnGroup:
    """Sequences in one attention call. They share a query length q, so
    their tokens form a contiguous [b, q] slice of the flat batch."""

    start: int  # first token of the group in the flat batch
    b: int
    q: int
    kv_slots: torch.Tensor | None  # [b, ctx] cache slots to read, None for a fresh prefill
    mask: torch.Tensor | None  # [b, 1, q, ctx] bool, True = may attend


@dataclass
class AttnBatch:
    slot_mapping: torch.Tensor  # [t] cache slot written by each new token
    groups: list


def load_weights(path, device, dtype):
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    w = {}
    for fn in files:
        with safe_open(fn, framework="pt") as f:
            for k in f.keys():
                w[k] = f.get_tensor(k).to(device=device, dtype=dtype)
    return w


class Model:
    def __init__(self, cfg: ModelConfig, weights: dict, device, dtype):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        p = "model."
        self.embed = weights[p + "embed_tokens.weight"]
        self.norm = weights[p + "norm.weight"]
        self.lm_head = self.embed if cfg.tie_embeddings else weights["lm_head.weight"]
        self.layers = []
        for i in range(cfg.num_layers):
            q = f"{p}layers.{i}."
            a = q + "self_attn."
            layer = {
                "ln1": weights[q + "input_layernorm.weight"],
                "ln2": weights[q + "post_attention_layernorm.weight"],
                # q, k and v share one matmul, and so do gate and up
                "qkv": torch.cat([weights[a + n + "_proj.weight"] for n in "qkv"]),
                "o": weights[a + "o_proj.weight"],
                "gate_up": torch.cat([weights[q + "mlp.gate_proj.weight"], weights[q + "mlp.up_proj.weight"]]),
                "down": weights[q + "mlp.down_proj.weight"],
                "qkv_b": torch.cat([weights[a + n + "_proj.bias"] for n in "qkv"]) if cfg.qkv_bias else None,
            }
            self.layers.append(layer)
        weights.clear()
        d = cfg.head_dim
        self.inv_freq = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64, device=device).float() / d)
        )
        self.scale = d**-0.5

    @classmethod
    def from_dir(cls, path, device="cpu", dtype=torch.float32):
        cfg = ModelConfig.from_dir(path)
        return cls(cfg, load_weights(path, device, dtype), device, dtype)

    def rms_norm(self, x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.cfg.rms_eps)
        return w * xf.to(x.dtype)

    def rope(self, positions):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self.dtype)[:, None, :], emb.sin().to(self.dtype)[:, None, :]

    @staticmethod
    def rotate(x, cos, sin):
        h = x.shape[-1] // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
        return x * cos + rot * sin

    def attention(self, q, k, v, k_cache, v_cache, batch: AttnBatch):
        out = torch.empty_like(q)
        for g in batch.groups:
            s, e = g.start, g.start + g.b * g.q
            qg = q[s:e].view(g.b, g.q, *q.shape[1:]).transpose(1, 2)
            if g.kv_slots is None:
                # whole prompt is new: attend to the fresh keys, causal
                kg = k[s:e].view(g.b, g.q, *k.shape[1:]).transpose(1, 2)
                vg = v[s:e].view(g.b, g.q, *v.shape[1:]).transpose(1, 2)
                o = F.scaled_dot_product_attention(qg, kg, vg, is_causal=True, scale=self.scale, enable_gqa=True)
            else:
                kg = k_cache[g.kv_slots].transpose(1, 2)  # [b, kvh, ctx, d]
                vg = v_cache[g.kv_slots].transpose(1, 2)
                o = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=g.mask, scale=self.scale, enable_gqa=True)
            out[s:e] = o.transpose(1, 2).reshape(e - s, *q.shape[1:])
        return out

    @torch.inference_mode()
    def forward(self, input_ids, positions, batch: AttnBatch, kv_cache, logit_index):
        """Runs the new tokens, fills their KV slots and returns float32
        logits for the rows in logit_index."""
        cfg = self.cfg
        t = input_ids.shape[0]
        nh, nkv, d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        cos, sin = self.rope(positions)
        x = self.embed[input_ids]
        for i, L in enumerate(self.layers):
            h = self.rms_norm(x, L["ln1"])
            qkv = F.linear(h, L["qkv"], L["qkv_b"])
            q, k, v = qkv.split([nh * d, nkv * d, nkv * d], dim=-1)
            q = self.rotate(q.view(t, nh, d), cos, sin)
            k = self.rotate(k.view(t, nkv, d), cos, sin)
            v = v.view(t, nkv, d)
            k_cache, v_cache = kv_cache[i, 0], kv_cache[i, 1]
            k_cache.index_copy_(0, batch.slot_mapping, k)
            v_cache.index_copy_(0, batch.slot_mapping, v)
            o = self.attention(q, k, v, k_cache, v_cache, batch)
            x = x + F.linear(o.view(t, nh * d), L["o"])
            h = self.rms_norm(x, L["ln2"])
            gate, up = F.linear(h, L["gate_up"]).chunk(2, dim=-1)
            x = x + F.linear(F.silu(gate) * up, L["down"])
        x = self.rms_norm(x[logit_index], self.norm)
        return F.linear(x, self.lm_head).float()

    def kv_bytes_per_token(self):
        c = self.cfg
        return 2 * c.num_layers * c.num_kv_heads * c.head_dim * torch.finfo(self.dtype).bits // 8

    def new_kv_cache(self, num_blocks, block_size):
        c = self.cfg
        return torch.zeros(
            c.num_layers, 2, num_blocks * block_size, c.num_kv_heads, c.head_dim, device=self.device, dtype=self.dtype
        )
