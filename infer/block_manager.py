"""Paged KV cache bookkeeping and prefix caching. Pure Python, no tensors.

The KV cache is a pool of fixed-size blocks (block_size token slots each).
A sequence owns a block table: logical block i of the sequence lives in
physical block table[i]. Blocks are allocated as the sequence grows, so
memory is reserved for tokens that exist, not for the longest possible
output.

Prefix caching: once a block is full and every token in it is final, it
is registered under a hash of (hash of the previous block, its tokens).
That key identifies the whole prefix up to and including the block. A new
sequence whose prompt starts with the same tokens finds those blocks and
reuses them instead of recomputing. Full blocks are never written again,
so sharing them needs no copy-on-write. A cached block nobody uses keeps
its contents and goes on an LRU list; it is only overwritten when the free
list runs dry.
"""

from collections import OrderedDict, deque


class KVState:
    """One sequence's view of one KV cache (the target and the draft model
    each have their own)."""

    __slots__ = ("block_table", "num_computed", "hashes")

    def __init__(self):
        self.block_table = []
        self.num_computed = 0  # tokens whose keys and values are in the cache
        self.hashes = []  # chained hash of each registered full block

    def reset(self):
        self.block_table = []
        self.num_computed = 0
        self.hashes = []


class BlockManager:
    def __init__(self, num_blocks, block_size, prefix_caching=True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.prefix_caching = prefix_caching
        self.ref = [0] * num_blocks
        self.hash_of = [None] * num_blocks
        self.cached = {}  # hash -> block
        self.free_list = deque(range(num_blocks))
        self.evictable = OrderedDict()  # cached blocks with ref 0, oldest first

    def num_free(self):
        return len(self.free_list) + len(self.evictable)

    def usage(self):
        return 1.0 - self.num_free() / self.num_blocks

    def blocks_for(self, num_tokens):
        return -(-num_tokens // self.block_size)

    def _take(self):
        if self.free_list:
            b = self.free_list.popleft()
        else:
            b, _ = self.evictable.popitem(last=False)
            del self.cached[self.hash_of[b]]
            self.hash_of[b] = None
        self.ref[b] = 1
        return b

    def _release(self, b):
        self.ref[b] -= 1
        assert self.ref[b] >= 0
        if self.ref[b] == 0:
            if self.hash_of[b] is not None:
                self.evictable[b] = None
            else:
                self.free_list.append(b)

    def _hash(self, parent, tokens):
        return hash((parent, tuple(tokens)))

    def match_prefix(self, st: KVState, ids):
        """Attaches cached blocks for the longest cached prefix of ids to an
        empty state. At least one token is always left to compute, since
        the last prompt token must run to produce logits."""
        assert not st.block_table
        if not self.prefix_caching:
            return 0
        bs = self.block_size
        parent = None
        hit = []
        for i in range((len(ids) - 1) // bs):
            h = self._hash(parent, ids[i * bs : (i + 1) * bs])
            b = self.cached.get(h)
            if b is None:
                break
            hit.append((b, h))
            parent = h
        for b, h in hit:
            if self.ref[b] == 0:
                del self.evictable[b]
            self.ref[b] += 1
            st.block_table.append(b)
            st.hashes.append(h)
        st.num_computed = len(hit) * bs
        return st.num_computed

    def can_ensure(self, st: KVState, num_tokens, reserve=0):
        return self.blocks_for(num_tokens) - len(st.block_table) <= self.num_free() - reserve

    def ensure(self, st: KVState, num_tokens, reserve=0):
        """Grows the block table to hold num_tokens. All or nothing."""
        if not self.can_ensure(st, num_tokens, reserve):
            return False
        while len(st.block_table) < self.blocks_for(num_tokens):
            st.block_table.append(self._take())
        return True

    def commit(self, st: KVState, ids):
        """Registers every full block whose tokens are all computed."""
        if not self.prefix_caching:
            return
        bs = self.block_size
        full = min(st.num_computed, len(ids)) // bs
        while len(st.hashes) < full:
            i = len(st.hashes)
            parent = st.hashes[-1] if st.hashes else None
            h = self._hash(parent, ids[i * bs : (i + 1) * bs])
            st.hashes.append(h)
            b = st.block_table[i]
            # another sequence may have cached the same prefix first; then
            # this copy stays private and is freed normally
            if h not in self.cached and self.hash_of[b] is None:
                self.cached[h] = b
                self.hash_of[b] = h

    def trim(self, st: KVState, num_tokens):
        """Frees blocks past num_tokens (rejected speculative tokens)."""
        keep = self.blocks_for(num_tokens)
        while len(st.block_table) > keep:
            b = st.block_table.pop()
            assert self.hash_of[b] is None  # only final tokens are ever hashed
            self._release(b)
        del st.hashes[keep:]

    def free(self, st: KVState):
        # last block first, so a prompt's tail is evicted before its head
        for b in reversed(st.block_table):
            self._release(b)
        st.reset()

    def check(self):
        """Invariants, for tests."""
        free = set(self.free_list)
        ev = set(self.evictable)
        assert len(free) == len(self.free_list)
        assert not (free & ev)
        for b in range(self.num_blocks):
            if b in free:
                assert self.ref[b] == 0 and self.hash_of[b] is None
            elif b in ev:
                assert self.ref[b] == 0 and self.cached[self.hash_of[b]] == b
            else:
                assert self.ref[b] > 0
        for h, b in self.cached.items():
            assert self.hash_of[b] == h
