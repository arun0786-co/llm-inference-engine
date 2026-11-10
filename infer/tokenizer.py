"""Text in and out: the checkpoint's tokenizer.json and chat template."""

import json
import os

from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, path):
        self.tok = HFTokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.template = None
        cfg = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                self.template = json.load(f).get("chat_template")

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def chat(self, messages):
        """Renders a list of {role, content} with the model's chat template
        and returns token ids, ending with the assistant turn's opening."""
        if not self.template:
            raise ValueError("this checkpoint has no chat template")
        import jinja2

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        text = env.from_string(self.template).render(messages=messages, add_generation_prompt=True)
        return self.encode(text)


class Detokenizer:
    """Turns a growing list of ids into text deltas for streaming. Holds
    back output that ends in a partial UTF-8 character."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.ids = []
        self.sent = ""

    def add(self, new_ids):
        self.ids.extend(new_ids)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta = text[len(self.sent) :]
        self.sent = text
        return delta
