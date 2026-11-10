"""OpenAI-compatible HTTP server.

One background thread owns the engine and calls step() in a loop. HTTP
handler threads hand requests to it through a queue and read tokens back
from a per-request queue, so the engine itself never needs a lock.

    python -m infer.server --model Qwen/Qwen2.5-Coder-0.5B-Instruct --port 8000
    curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
"""

import argparse
import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import Engine
from .hub import resolve
from .sequence import SamplingParams
from .tokenizer import Detokenizer, Tokenizer


class EngineLoop:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.inbox = queue.Queue()
        self.outboxes = {}
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, prompt_ids, params):
        """Returns a queue that yields lists of new token ids, then the
        finished Sequence, or an Exception."""
        out = queue.Queue()
        self.inbox.put(("add", prompt_ids, params, out))
        return out

    def abort(self, out):
        self.inbox.put(("abort", out))

    def _run(self):
        eng = self.engine
        while True:
            block = not eng.has_work()
            try:
                while True:
                    msg = self.inbox.get(block=block, timeout=None)
                    block = False
                    if msg[0] == "add":
                        _, ids, params, out = msg
                        try:
                            self.outboxes[eng.add_request(ids, params)] = out
                        except ValueError as e:
                            out.put(e)
                    else:
                        for seq, out in list(self.outboxes.items()):
                            if out is msg[1]:
                                eng.abort(seq)
                                del self.outboxes[seq]
            except queue.Empty:
                pass
            try:
                events = eng.step()
            except Exception as e:  # fail the requests in flight, keep serving
                for seq, out in self.outboxes.items():
                    eng.abort(seq)
                    out.put(RuntimeError(f"engine error: {e}"))
                self.outboxes.clear()
                continue
            for seq, toks in events:
                out = self.outboxes.get(seq)
                if out is None:
                    continue
                out.put(toks)
                if seq.finished:
                    out.put(seq)
                    del self.outboxes[seq]


def params_from(body):
    return SamplingParams(
        temperature=float(body.get("temperature", 1.0)),
        top_p=float(body.get("top_p", 1.0)),
        top_k=int(body.get("top_k", 0)),
        max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 256),
        seed=body.get("seed"),
        ignore_eos=bool(body.get("ignore_eos", False)),
    )


def make_handler(loop: EngineLoop, tok: Tokenizer, model_name):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            elif self.path == "/health":
                self._json(200, {"ok": True})
            elif self.path == "/metrics":
                self._json(200, loop.engine.stats())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/v1/chat/completions":
                    chat = True
                    ids = tok.chat(body["messages"])
                elif self.path == "/v1/completions":
                    chat = False
                    p = body["prompt"]
                    ids = p if isinstance(p, list) else tok.encode(p)
                else:
                    return self._json(404, {"error": "not found"})
                params = params_from(body)
            except (KeyError, ValueError, TypeError) as e:
                return self._json(400, {"error": f"bad request: {e}"})
            out = loop.submit(ids, params)
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:16]
            if body.get("stream"):
                self._stream(out, rid, chat)
            else:
                self._whole(out, rid, chat, len(ids))

        def _whole(self, out, rid, chat, n_prompt):
            ids = []
            while True:
                m = out.get()
                if isinstance(m, Exception):
                    return self._json(400, {"error": str(m)})
                if isinstance(m, list):
                    ids.extend(m)
                    continue
                seq = m
                break
            text = tok.decode(ids)
            choice = {"index": 0, "finish_reason": seq.finish_reason}
            if chat:
                choice["message"] = {"role": "assistant", "content": text}
            else:
                choice["text"] = text
            self._json(
                200,
                {
                    "id": rid,
                    "object": "chat.completion" if chat else "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [choice],
                    "usage": {
                        "prompt_tokens": n_prompt,
                        "completion_tokens": len(ids),
                        "total_tokens": n_prompt + len(ids),
                        "cached_prompt_tokens": seq.cached_tokens,
                    },
                },
            )

        def _stream(self, out, rid, chat):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            detok = Detokenizer(tok)

            def send(obj):
                data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

            def chunk(text, finish=None):
                if chat:
                    c = {"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}
                else:
                    c = {"index": 0, "text": text, "finish_reason": finish}
                return {
                    "id": rid,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [c],
                }

            try:
                while True:
                    m = out.get()
                    if isinstance(m, Exception):
                        send({"error": str(m)})
                        break
                    if isinstance(m, list):
                        delta = detok.add(m)
                        if delta:
                            send(chunk(delta))
                        continue
                    send(chunk("", m.finish_reason))
                    break
                send(b"[DONE]")
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                loop.abort(out)  # client went away; free its kv blocks

    return Handler


def serve(engine, tok, model_name, host="127.0.0.1", port=8000):
    loop = EngineLoop(engine)
    httpd = ThreadingHTTPServer((host, port), make_handler(loop, tok, model_name))
    return httpd, loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir or hugging face repo id")
    ap.add_argument("--draft", help="draft model for speculative decoding")
    ap.add_argument("--spec-tokens", type=int, default=4)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--kv-cache-gb", type=float, default=2.0)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--device")
    a = ap.parse_args()
    path = resolve(a.model)
    eng = Engine(
        path,
        device=a.device,
        kv_cache_gb=a.kv_cache_gb,
        max_num_seqs=a.max_num_seqs,
        draft_model_path=resolve(a.draft) if a.draft else None,
        num_spec_tokens=a.spec_tokens,
    )
    httpd, _ = serve(eng, Tokenizer(path), a.model, a.host, a.port)
    print(f"serving {a.model} on http://{a.host}:{httpd.server_address[1]} ({eng.device}, {eng.bm.num_blocks} kv blocks)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
