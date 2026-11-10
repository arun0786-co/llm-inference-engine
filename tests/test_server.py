"""The http server on a random checkpoint with a toy word-level tokenizer."""

import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest
from tokenizers import Tokenizer as HFTok
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from conftest import make_tiny
from infer.engine import Engine
from infer.sequence import SamplingParams
from infer.server import serve
from infer.tokenizer import Tokenizer
from test_engine import prompts


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("srv"))
    make_tiny(path, "llama")
    t = HFTok(WordLevel({f"t{i}": i for i in range(256)}, unk_token="t0"))
    t.pre_tokenizer = WhitespaceSplit()
    t.save(path + "/tokenizer.json")
    tok = Tokenizer(path)
    eng = Engine(path, device="cpu", num_blocks=256, block_size=4)
    httpd, _ = serve(eng, tok, "tiny", port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", path, tok
    httpd.shutdown()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode()


def test_concurrent_requests_match_offline(server):
    url, path, tok = server
    ps = prompts(8, 20)
    ref = Engine(path, device="cpu", num_blocks=256, block_size=4).generate(
        ps, SamplingParams(max_tokens=10, ignore_eos=True)
    )
    body = lambda p: {"prompt": p, "max_tokens": 10, "temperature": 0, "ignore_eos": True}
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda p: json.loads(post(url + "/v1/completions", body(p))), ps))
    for o, r in zip(outs, ref):
        assert o["choices"][0]["text"] == tok.decode(r)
        assert o["usage"]["completion_tokens"] == 10


def test_stream_equals_whole(server):
    url, _, _ = server
    body = {"prompt": "t5 t9 t33 t7", "max_tokens": 12, "temperature": 0.8, "seed": 3, "ignore_eos": True}
    whole = json.loads(post(url + "/v1/completions", body))["choices"][0]["text"]
    raw = post(url + "/v1/completions", body | {"stream": True})
    events = [line[6:] for line in raw.split("\n") if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e)["choices"][0] for e in events[:-1]]
    assert "".join(c["text"] for c in chunks) == whole
    assert chunks[-1]["finish_reason"] == "length"
    st = json.loads(urllib.request.urlopen(url + "/metrics").read())
    assert st["running"] == 0 and st["kv_usage"] == 0
