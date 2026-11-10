import random

from infer.block_manager import BlockManager, KVState


def test_prefix_sharing_and_lru():
    bm = BlockManager(8, 4)
    a, b = KVState(), KVState()
    ids = list(range(10))  # two full blocks and a partial one
    assert bm.ensure(a, 10)
    a.num_computed = 10
    bm.commit(a, ids)
    assert len(bm.cached) == 2
    # same prefix, longer prompt: both full blocks are reused
    assert bm.match_prefix(b, ids + [99, 98]) == 8
    assert b.block_table[:2] == a.block_table[:2]
    assert bm.ref[a.block_table[0]] == 2
    bm.free(a)
    bm.free(b)
    bm.check()
    assert bm.num_free() == 8 and len(bm.evictable) == 2
    # a prompt that is exactly the cached blocks still recomputes one block
    c = KVState()
    assert bm.match_prefix(c, ids[:8]) == 4


def test_eviction_reuses_oldest_cached_block():
    bm = BlockManager(3, 2)
    s = KVState()
    bm.ensure(s, 6)
    s.num_computed = 6
    bm.commit(s, [1, 2, 3, 4, 5, 6])
    bm.free(s)  # all three blocks cached and evictable
    t = KVState()
    assert bm.ensure(t, 2)  # evicts the tail block first
    assert bm.match_prefix(KVState(), [1, 2, 3, 4, 9]) == 4
    bm.check()


def test_random_operations_keep_invariants():
    """Random admit / grow / commit / trim / free on shared prefixes."""
    rng = random.Random(0)
    bm = BlockManager(64, 4)
    live = []
    prefixes = [[rng.randrange(50) for _ in range(rng.randint(0, 20))] for _ in range(4)]
    for _ in range(5000):
        op = rng.random()
        if op < 0.3 and len(live) < 12:
            ids = rng.choice(prefixes) + [rng.randrange(50) for _ in range(rng.randint(1, 10))]
            st = KVState()
            hit = bm.match_prefix(st, ids)
            assert hit % 4 == 0 and hit < len(ids)
            if bm.ensure(st, len(ids)):
                live.append((st, ids))
            else:
                bm.free(st)
        elif op < 0.6 and live:
            st, ids = rng.choice(live)
            grow = [rng.randrange(50) for _ in range(rng.randint(1, 5))]
            if bm.ensure(st, len(ids) + len(grow)):
                ids.extend(grow)
                st.num_computed = len(ids) - rng.randint(0, 1)
                bm.commit(st, ids)
        elif op < 0.75 and live:
            # rejected speculative tokens: drop a few uncomputed ones
            st, ids = rng.choice(live)
            del ids[max(st.num_computed, len(ids) - 3) + 1 :]
            bm.trim(st, len(ids))
            assert len(st.block_table) <= bm.blocks_for(len(ids))
        elif live:
            st, _ = live.pop(rng.randrange(len(live)))
            bm.free(st)
        bm.check()
        refs = sum(bm.ref)
        assert refs == sum(len(st.block_table) for st, _ in live)
    for st, _ in live:
        bm.free(st)
    bm.check()
    assert bm.num_free() == 64
