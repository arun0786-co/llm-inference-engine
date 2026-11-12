import pytest
import torch


def make_tiny(path, kind="llama", vocab=256, seed=0, layers=2, hidden=64):
    """Saves a small random checkpoint with transformers, so tests need no
    download and compare against the reference implementation."""
    import transformers

    torch.manual_seed(seed)
    common = dict(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        bos_token_id=1,
        eos_token_id=2,
    )
    if kind == "llama":
        cfg = transformers.LlamaConfig(**common, tie_word_embeddings=False)
        model = transformers.LlamaForCausalLM(cfg)
    else:
        cfg = transformers.Qwen2Config(**common, tie_word_embeddings=True)
        model = transformers.Qwen2ForCausalLM(cfg)
    # the default init (std 0.02) gives near-uniform logits; widen it so
    # comparisons and sampling tests see real structure
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() > 1:
                p.normal_(0, 0.15)
            elif name.endswith("bias"):
                p.normal_(0, 0.1)
    model.eval()
    model.save_pretrained(path)
    return model


@pytest.fixture(scope="session", params=["llama", "qwen2"])
def tiny(request, tmp_path_factory):
    path = tmp_path_factory.mktemp(request.param)
    model = make_tiny(str(path), request.param)
    return str(path), model
