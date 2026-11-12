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

from . import metal
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
    tables: torch.Tensor | None = None  # [b, blocks] int32, set when the metal kernel runs
    ctx_lens: torch.Tensor | None = None  # [b] int32


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
        self.lm_head_slices = 8 if str(device).startswith("mps") else 1
        self.block_size = None  # set by the runner
        on_mps = str(device).startswith("mps")
        self.paged_kernel = on_mps and metal.available(dtype, d, cfg.num_heads // cfg.num_kv_heads)

    @classmethod
    def from_dir(cls, path, device="cpu", dtype=torch.float32):
        cfg = ModelConfig.from_dir(path)
        return cls(cfg, load_weights(path, device, dtype), device, dtype)

    def rms_norm(self, x, w):
        # normalise in float32 and round once, as the reference does; the
        # fused kernel on a bf16 input rounds differently
        return w * F.rms_norm(x.float(), (x.shape[-1],), None, self.cfg.rms_eps).to(x.dtype)

    def rope(self, positions):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self.dtype)[:, None, :], emb.sin().to(self.dtype)[:, None, :]

    @staticmethod
    def rotate(x, cos, sin):
        h = x.shape[-1] // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
        return x * cos + rot * sin

    def attention(self, qkv, cache, batch: AttnBatch):
        nh, nkv = self.cfg.num_heads, self.cfg.num_kv_heads
        outs = []
        for g in batch.groups:
            x = qkv[g.start : g.start + g.b * g.q].view(g.b, g.q, *qkv.shape[1:])
            if g.tables is not None:
                o = metal.paged_decode(x[:, 0, :nh], cache, g.tables, g.ctx_lens, nkv, self.block_size, self.scale)
                outs.append(o.view(g.b, nh * self.cfg.head_dim))
                continue
            q = x[:, :, :nh].transpose(1, 2)
            if g.kv_slots is None:
                # whole prompt is new: attend to the fresh keys, causal
                kv, causal = x, True
            else:
                # index_select is ~10x faster than cache[kv_slots] on mps
                kv = cache.index_select(0, g.kv_slots.view(-1)).view(*g.kv_slots.shape, *cache.shape[1:])
                causal = False  # kv: [b, ctx, 2 * kvh, d]
            off = nh if causal else 0  # fresh rows hold q, k, v heads; cache rows hold k, v
            k = kv[:, :, off : off + nkv].transpose(1, 2)
            v = kv[:, :, off + nkv : off + 2 * nkv].transpose(1, 2)
            o = F.scaled_dot_product_attention(
                q, k, v, attn_mask=g.mask, is_causal=causal, scale=self.scale, enable_gqa=True
            )
            outs.append(o.transpose(1, 2).reshape(g.b * g.q, nh * self.cfg.head_dim))
        return outs[0] if len(outs) == 1 else torch.cat(outs)

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
            qkv = F.linear(h, L["qkv"], L["qkv_b"]).view(t, nh + 2 * nkv, d)
            # rotate q and k heads together, in place
            qkv[:, : nh + nkv] = self.rotate(qkv[:, : nh + nkv], cos, sin)
            cache = kv_cache[i]
            cache.index_copy_(0, batch.slot_mapping, qkv[:, nh:])
            o = self.attention(qkv, cache, batch)
            x = x + F.linear(o, L["o"])
            h = self.rms_norm(x, L["ln2"])
            gate, up = F.linear(h, L["gate_up"]).chunk(2, dim=-1)
            x = x + F.linear(F.silu(gate) * up, L["down"])
        x = self.rms_norm(x[logit_index], self.norm)
        return self.logits(x)

    def logits(self, x):
        if self.lm_head_slices == 1 or x.shape[0] < 16:
            return F.linear(x, self.lm_head).float()
        # mps picks a slow kernel for [b >= 16, h] x [h, 150k]; eight vocab
        # slices are 2.2x faster at b = 64
        return torch.cat([F.linear(x, w) for w in self.lm_head.chunk(self.lm_head_slices)], dim=-1).float()

    def kv_bytes_per_token(self):
        c = self.cfg
        return 2 * c.num_layers * c.num_kv_heads * c.head_dim * torch.finfo(self.dtype).bits // 8

    def new_kv_cache(self, num_blocks, block_size):
        c = self.cfg
        # one row per token slot: its k heads, then its v heads
        return torch.zeros(
            c.num_layers, num_blocks * block_size, 2 * c.num_kv_heads, c.head_dim, device=self.device, dtype=self.dtype
        )
