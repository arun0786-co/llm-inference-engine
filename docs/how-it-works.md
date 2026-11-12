# How it works

These notes walk through the engine in the order a request meets it. Each section names the file that implements it.

## 1. What an inference server has to solve

A decoder-only transformer generates one token per forward pass. Each pass needs the keys and values of every earlier token at every layer, so they are cached (the KV cache) instead of recomputed. For Qwen2.5-0.5B that is 12 KB per token in bf16 (24 layers × 2 kv heads × 64 dims × 2 for k and v × 2 bytes). A 1,000-token conversation therefore holds 12 MB.

One decode step for one sequence reads all the weights (about 1 GB for the 0.5B model) to produce one token, which leaves the GPU mostly idle on arithmetic. Running 64 sequences in the same pass reads the weights once for 64 tokens. Serving throughput is mostly about keeping many sequences in the batch, and that is limited by KV cache memory and by how the batch is managed.

## 2. Paged KV cache (`block_manager.py`, `model.py`)

The simple design reserves `max_length` slots per request up front. Most requests stop far earlier, so most of that memory is reserved and never used. vLLM's idea (Kwon et al., 2023, "PagedAttention") is to treat the cache like virtual memory:

- The cache is one big tensor of slots, `[layers, num_blocks * block_size, 2 * kv_heads, head_dim]`. Each row holds one token's k heads, then its v heads.
- Slots are grouped into blocks of 16 tokens. A sequence owns a block table: logical block i is physical block `table[i]`.
- Token at position p lives in slot `table[p // 16] * 16 + p % 16`.
- A new block is taken from the free list only when the sequence crosses a block boundary, so at most 15 slots per sequence are wasted.

Attention then has to read a sequence's keys from scattered blocks. The portable path (`Model.attention`) builds the slot index of every context position and uses `index_select` to gather them into a padded `[batch, max_ctx]` tensor, then runs `scaled_dot_product_attention` with a mask. The Metal kernel (section 8) reads the blocks directly.

## 3. Prefix caching (`block_manager.py`)

Many requests share a beginning: a system prompt, a few-shot block, a document under discussion. Once a block is full and all its tokens are final, it is registered under `hash((hash of previous block, its 16 tokens))`. Because the key chains through the previous block, it identifies the entire prefix, not just these 16 tokens.

A new request walks its prompt block by block and stops at the first miss. Every hit is a block whose keys and values are already computed, so it is attached with a reference count instead of being recomputed.

- **No copy-on-write.** Full blocks are never written again, and only full blocks are shared, so sharing is safe without copies.
- **Freed blocks stay cached.** A cached block nobody is using goes to an LRU list. It keeps its contents until the free list is empty and the block is needed for something else.
- **One token always runs.** If the whole prompt hits, the last block is still recomputed, because the model has to run the last prompt token to produce logits for the first output token.

## 4. Continuous batching, chunked prefill, preemption (`scheduler.py`)

Static batching (what `transformers.generate` does with a list of prompts) pads the batch to the longest prompt and runs until the longest output finishes. A request that wanted 16 tokens sits in the batch for 512 steps.

Continuous batching (Yu et al., 2022, "Orca") schedules per step instead of per batch. Every step the scheduler:

1. Gives each running sequence its next work. That is one token if it is decoding (1 + k with speculative decoding), or the next chunk of its prompt if it is still prefilling.
2. Fills the remaining token budget (`max_batched_tokens`) with waiting requests, oldest first, up to `max_num_seqs` running.
3. Lets a finished sequence leave at once and frees its blocks for the next request.

**Chunked prefill.** A 2,000-token prompt need not run in one step. It takes whatever budget is left and continues next step. This keeps step time bounded, so sequences already decoding are not stalled behind a big prompt.

**Preemption.** If blocks run out mid-flight, the newest running sequence is preempted:
- its blocks are freed and it goes back to the front of the queue;
- when it returns, its prompt plus the tokens it already produced are recomputed, mostly from the prefix cache.

Its random number stream continues where it was, so preemption does not change its output (`test_preemption_does_not_change_output`).

## 5. One forward pass for a mixed batch (`runner.py`, `model.py`)

A step can contain sequences bringing 1 token, 7 tokens, 512 tokens. The runner lays all new tokens out flat, grouped by how many tokens each sequence brings. Each group is then a contiguous `[b, q]` slice and needs one attention call:
- **Fresh prompts** (nothing cached yet) use causal attention on the new keys directly.
- **Every other group** reads its context from the cache, with a mask that lets query r of a sequence see keys up to position `ctx - q + r`.

The matmuls, norms and MLPs do not care about sequence boundaries and run once over all tokens.

The model code is about 150 lines written against the checkpoint tensors. q, k and v are one matmul, and so are gate and up. The model is compared with transformers on random checkpoints: the logits match exactly in float32 on CPU, for prefill, chunked prefill, decode and mixed batches (`tests/test_model.py`).

## 6. Sampling (`sampler.py`)

Temperature, top-k and top-p turn logits into a distribution. The draw itself inverts the CDF with one uniform number from the request's own `random.Random(seed)`: token i is chosen when `cdf[i-1] <= u < cdf[i]`. That needs one float per request per token. The output depends only on the request's seed and its logits, not on which other requests share the batch (`test_sampling_is_per_request`).

## 7. Speculative decoding (`engine.py: _spec_step`)

Decoding at batch 1 is memory bound: reading 3 GB of weights to produce one token. A forward pass over 5 tokens costs about the same as over 1, so if the next few tokens could be guessed, one target pass could check all of them.

Each pass works like this (Leviathan et al., 2023; Chen et al., 2023):

1. A small draft model (same tokenizer) proposes k tokens d1..dk, one at a time. It keeps its own paged cache.
2. The target model runs `[last token, d1, ..., dk]` in one pass, which gives its distribution p at all k+1 positions.
3. Walk left to right. Accept di with probability `min(1, p(di) / q(di))`, where q is the draft's distribution. At the first rejection, sample a replacement from `max(0, p - q)` renormalised and stop. If all k are accepted, sample one more token from the target's last position for free.

**Why the output is exactly the target's distribution.** Take a single position. The token comes out as x in two ways:
- the draft proposed x and it was accepted: probability `q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))`;
- something was rejected and x was drawn from the residual: probability `(1 - Σ min(q, p)) · max(0, p(x) - q(x)) / Σ max(0, p - q)`.

Since `Σ max(0, p - q) = 1 - Σ min(p, q)`, the second term is `max(0, p(x) - q(x))`. The sum is `min(p, q) + max(0, p - q) = p(x)`. The same argument applies at each later position, conditioned on the prefix.

With greedy decoding this reduces to "accept while the target's argmax agrees", and the output must equal the target's own greedy output.

**Bookkeeping.** After a pass the target cache holds k+1 new tokens, of which only the accepted ones are valid, so the sequence's computed length is set back and blocks past it are freed. The draft cache is trimmed the same way. A draft that is behind (a new request, or all k accepted) catches up at the start of the next pass by running all tokens it has not seen, using the same mixed-batch path as prefill.

**Tests** (`tests/test_spec.py`):
- greedy output equals the target's for k = 1, 3, 5;
- the target used as its own draft accepts everything;
- a preemption in the middle of speculation does not change the output;
- with a vocabulary of 4 and temperature 1, 12,000 three-token samples follow the target's exact joint distribution: chi-square 45.9 on 63 degrees of freedom, where the 99.9% limit is 103.4. The draft's own distribution scores 131,000.

A deliberately wrong version that resamples from p instead of `max(0, p - q)` fails with chi-square 3,383.

## 8. The Metal kernel (`metal.py`)

On Apple GPUs, gather + SDPA has two costs:
- the gather copies every sequence's whole context once per layer;
- padding makes every sequence as long as the longest in the batch.

At 256 sequences with contexts between 100 and 1,000 tokens, it took 3.2 ms per layer, 24 layers, every step.

The kernel computes decode attention straight from the block table, with one threadgroup per (sequence, kv head):
- **One read per key.** The kv head's query heads are handled together (grouped-query attention: 7 query heads share each kv head in Qwen2.5-0.5B).
- **Scores.** Four simdgroups take turns on chunks of 32 keys. Lane j of a simdgroup reads key j and computes its scores for all query heads, so 32 key reads are in flight together.
- **Online softmax.** `simd_max` and `simd_sum` over the chunk update a running max and sum per head (Milakov and Gimelshein, 2018). The whole score row is never stored.
- **Values.** The simdgroup switches layout: 8 lanes share one value row (8 dims each, two vector loads), so 4 value rows are read at once. Lanes holding the same dims are summed with `simd_shuffle_xor` at the end.
- **Merge.** The four simdgroups' (max, sum, accumulator) triples are combined through threadgroup memory.

It is compiled at run time with `torch.mps.compile_shader` and matches the reference to 4e-7 in float32 (`tests/test_metal.py`).

How the design got there (bf16, 64 sequences with contexts 100 to 1,000, then 256 sequences):

| version | 64 seqs | 256 seqs |
|---|---|---|
| gather + sdpa | 0.93 ms | 3.24 ms |
| one simdgroup per key stream, simd_sum per key | 1.02 ms | 3.36 ms |
| lane per key for scores, lane per dim for values | 0.69 ms | 1.84 ms |
| plus 4 value rows in flight per simdgroup | 0.45 ms | 1.25 ms |

The first version was slower than the baseline because each simdgroup had a single 128-byte load in flight at a time. At batch 1 the kernel launches only 2 threadgroups and loses to SDPA (0.25 vs 0.12 ms), so the runner uses it only for decode batches of 8 or more.

## 9. Two things specific to PyTorch on MPS

Both were found by profiling.
- **Gathering.** `cache[index]` (advanced indexing) gathers about 10x slower than `cache.index_select(0, index)`: 0.39 ms vs 0.036 ms for 64 × 200 rows.
- **The vocabulary projection.** For `[b, 896] × [896, 151936]`, MPS picks a much slower matmul kernel once b reaches 16 (1.2 ms at b = 8, 5.3 ms at b = 16). Splitting the vocabulary into 8 slices and concatenating takes 3.5 ms at b = 64 instead of 7.7 ms.

## 10. Server (`server.py`)

`ThreadingHTTPServer` handles the connections. One background thread owns the engine and loops on `step()`. Handler threads put requests on a queue and read their tokens back from a per-request queue, so the engine needs no lock. Streaming uses server-sent events in the OpenAI chunk format. The detokenizer holds back text ending in a partial UTF-8 character. A client that disconnects is aborted, and its blocks are freed.

## References

- W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. https://arxiv.org/abs/2309.06180
- G. Yu et al. Orca: A Distributed Serving System for Transformer-Based Generative Models. OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu
- Y. Leviathan, M. Kalman, Y. Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023. https://arxiv.org/abs/2211.17192
- C. Chen et al. Accelerating Large Language Model Decoding with Speculative Sampling. 2023. https://arxiv.org/abs/2302.01318
- M. Milakov, N. Gimelshein. Online normalizer calculation for softmax. 2018. https://arxiv.org/abs/1805.02867
- A. Agrawal et al. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (chunked prefill). OSDI 2024. https://arxiv.org/abs/2403.02310
