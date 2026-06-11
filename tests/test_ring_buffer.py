"""T1.1 AC: wraparound, overflow policy, blocking timeout, producer/consumer
soak with zero data corruption."""
import threading
import time

import numpy as np

from avfusion.audio.ring_buffer import RingBuffer


def _hop(value, hop=4, ch=2):
    return np.full((hop, ch), value, dtype=np.int16)


def test_fifo_order_across_wraparound():
    rb = RingBuffer(hop_samples=4, channels=2, capacity_hops=8)
    for i in range(20):                      # wraps the 8-slot buffer twice
        rb.push(_hop(i % 100), pts=float(i))
        hop, pts = rb.pop(timeout=0.1)
        assert hop[0, 0] == i % 100
        assert pts == float(i)
    assert rb.dropped == 0


def test_overflow_drops_oldest_and_counts():
    rb = RingBuffer(hop_samples=4, channels=2, capacity_hops=4)
    for i in range(6):
        rb.push(_hop(i), pts=float(i))
    assert rb.dropped == 2
    got = [rb.pop(timeout=0.1)[0][0, 0] for _ in range(4)]
    assert got == [2, 3, 4, 5]               # oldest two were sacrificed
    assert rb.pop(timeout=0.05) is None


def test_pop_blocks_until_timeout():
    rb = RingBuffer(hop_samples=4, channels=2, capacity_hops=4)
    t0 = time.monotonic()
    assert rb.pop(timeout=0.2) is None
    assert time.monotonic() - t0 >= 0.19


def test_close_unblocks_consumer():
    rb = RingBuffer(hop_samples=4, channels=2, capacity_hops=4)
    out = {}

    def consumer():
        out["item"] = rb.pop(timeout=5.0)

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.1)
    rb.close()
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert out["item"] is None


def test_push_rejects_wrong_shape():
    rb = RingBuffer(hop_samples=4, channels=2, capacity_hops=4)
    try:
        rb.push(np.zeros((3, 2), dtype=np.int16), pts=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_soak_threaded_no_corruption():
    """10^6 frames through producer/consumer threads. Sequence numbers are
    embedded in the payload; drop-oldest preserves order, so the consumer
    must observe a strictly increasing subsequence, and observed + dropped
    must account for every frame."""
    n = 1_000_000
    rb = RingBuffer(hop_samples=2, channels=1, capacity_hops=4096,
                    dtype=np.int64)
    seen = []

    def producer():
        hop = np.empty((2, 1), dtype=np.int64)
        for i in range(n):
            hop[:, 0] = i
            rb.push(hop, pts=float(i))
        rb.close()

    def consumer():
        while True:
            item = rb.pop(timeout=2.0)
            if item is None:
                return
            hop, pts = item
            assert hop[0, 0] == hop[1, 0] == int(pts)   # intact payload
            seen.append(int(hop[0, 0]))

    tp = threading.Thread(target=producer)
    tc = threading.Thread(target=consumer)
    tp.start(); tc.start()
    tp.join(timeout=120); tc.join(timeout=120)
    assert not tp.is_alive() and not tc.is_alive()

    assert all(b > a for a, b in zip(seen, seen[1:]))    # order preserved
    assert len(seen) + rb.dropped == n                   # nothing unaccounted
    assert seen[-1] == n - 1                             # freshest data won
