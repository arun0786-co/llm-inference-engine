import os


def resolve(name):
    """A local checkpoint directory, or a Hugging Face repo id to fetch."""
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download

    return snapshot_download(
        name, allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer.model"]
    )
