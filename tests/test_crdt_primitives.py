"""
test_crdt_primitives.py — Property-based tests for CRDT data structures.

Tests mathematical invariants of core CRDT types:
- Commutativity: merge(A, B) == merge(B, A)
- Associativity: merge(merge(A, B), C) == merge(A, merge(B, C))
- Idempotency: merge(A, A) == A
- Monotonicity: value never decreases across merges
"""

import time
import json
from smartcrdt_fleet.crdt_primitives import (
    GCounter, PNCounter, LWWRegister, ORSet, VectorClock,
    HybridLogicalClock, HLCTimestamp,
)


# ===========================================================================
# VectorClock Tests
# ===========================================================================

class TestVectorClock:
    def test_increment_increases_counter(self):
        vc = VectorClock("r1")
        assert vc.get("r1") == 0
        vc.increment()
        assert vc.get("r1") == 1
        vc.increment()
        assert vc.get("r1") == 2

    def test_merge_elementwise_max(self):
        vc1 = VectorClock("r1", {"r1": 3, "r2": 1})
        vc2 = VectorClock("r2", {"r1": 1, "r2": 5, "r3": 2})
        merged = vc1.merge(vc2)
        assert merged.get("r1") == 3
        assert merged.get("r2") == 5
        assert merged.get("r3") == 2

    def test_commutativity(self):
        vc1 = VectorClock("r1", {"r1": 5, "r2": 2})
        vc2 = VectorClock("r2", {"r1": 1, "r2": 7})
        m1 = vc1.merge(vc2)
        m2 = vc2.merge(vc1)
        # Compare counters (replica_id differs)
        assert m1.to_dict()["counters"] == m2.to_dict()["counters"]

    def test_associativity(self):
        vc1 = VectorClock("r1", {"r1": 1})
        vc2 = VectorClock("r2", {"r2": 2})
        vc3 = VectorClock("r3", {"r3": 3})
        left = vc1.merge(vc2).merge(vc3)
        right = vc1.merge(vc2.merge(vc3))
        assert left.to_dict() == right.to_dict()

    def test_idempotency(self):
        vc = VectorClock("r1", {"r1": 5, "r2": 3})
        merged = vc.merge(vc)
        assert merged.to_dict()["counters"] == vc.to_dict()["counters"]

    def test_happens_before(self):
        vc1 = VectorClock("r1", {"r1": 2, "r2": 1})
        vc2 = VectorClock("r2", {"r1": 2, "r2": 3})
        assert vc1.happens_before(vc2)
        assert not vc2.happens_before(vc1)

    def test_concurrent(self):
        vc1 = VectorClock("r1", {"r1": 3, "r2": 1})
        vc2 = VectorClock("r2", {"r1": 1, "r2": 3})
        assert vc1.is_concurrent(vc2)
        assert vc2.is_concurrent(vc1)

    def test_serialization_roundtrip(self):
        vc = VectorClock("r1", {"r1": 5, "r2": 3, "r3": 7})
        data = vc.to_dict()
        restored = VectorClock.from_dict(data)
        assert restored == vc

    def test_empty_clocks(self):
        vc1 = VectorClock("r1")
        vc2 = VectorClock("r2")
        merged = vc1.merge(vc2)
        assert merged.get("r1") == 0
        assert merged.get("r2") == 0


# ===========================================================================
# HLC Tests
# ===========================================================================

class TestHybridLogicalClock:
    def test_send_increments(self):
        hlc = HybridLogicalClock("n1")
        t1 = hlc.send_event()
        t2 = hlc.send_event()
        assert t2 >= t1

    def test_receive_advances(self):
        hlc1 = HybridLogicalClock("n1")
        hlc2 = HybridLogicalClock("n2")
        t1 = hlc1.send_event()
        t2 = hlc2.receive_event(t1)
        assert t2.physical_time >= t1.physical_time

    def test_physical_time_tracks(self):
        hlc = HybridLogicalClock("n1")
        before = time.time() * 1000
        ts = hlc.now()
        after = time.time() * 1000
        assert before - 100 <= ts.physical_time <= after + 100

    def test_serialization_roundtrip(self):
        hlc = HybridLogicalClock("n1", epsilon_ms=5)
        data = hlc.to_dict()
        restored = HybridLogicalClock.from_dict(data)
        assert restored._node_id == "n1"
        assert restored._epsilon == 5

    def test_hlc_timestamp_ordering(self):
        t1 = HLCTimestamp(physical_time=1000, logical_counter=0, node_id="a")
        t2 = HLCTimestamp(physical_time=1000, logical_counter=1, node_id="a")
        t3 = HLCTimestamp(physical_time=1001, logical_counter=0, node_id="a")
        assert t1 < t2 < t3

    def test_hlc_timestamp_tiebreak(self):
        t1 = HLCTimestamp(physical_time=1000, logical_counter=0, node_id="a")
        t2 = HLCTimestamp(physical_time=1000, logical_counter=0, node_id="b")
        assert t1 != t2
        assert (t1 < t2) != (t2 < t1)  # deterministic ordering


# ===========================================================================
# G-Counter Tests
# ===========================================================================

class TestGCounter:
    def test_increment(self):
        gc = GCounter("r1")
        gc.increment()
        gc.increment(5)
        assert gc.value() == 6

    def test_negative_increment_raises(self):
        gc = GCounter("r1")
        try:
            gc.increment(-1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_merge_max(self):
        gc1 = GCounter("r1", {"r1": 5, "r2": 3})
        gc2 = GCounter("r2", {"r1": 2, "r2": 7, "r3": 1})
        merged = gc1.merge(gc2)
        assert merged.value() == 5 + 7 + 1  # max per replica, then sum

    def test_commutativity(self):
        gc1 = GCounter("r1", {"r1": 3, "r2": 5})
        gc2 = GCounter("r2", {"r1": 7, "r2": 2})
        assert gc1.merge(gc2).value() == gc2.merge(gc1).value()

    def test_associativity(self):
        gc1 = GCounter("r1", {"r1": 1})
        gc2 = GCounter("r2", {"r2": 2})
        gc3 = GCounter("r3", {"r3": 3})
        left = gc1.merge(gc2).merge(gc3)
        right = gc1.merge(gc2.merge(gc3))
        assert left.value() == right.value()

    def test_idempotency(self):
        gc = GCounter("r1", {"r1": 5})
        assert gc.merge(gc).value() == gc.value()

    def test_monotonicity(self):
        gc1 = GCounter("r1", {"r1": 3})
        gc2 = GCounter("r2", {"r2": 4})
        merged = gc1.merge(gc2)
        assert merged.value() >= gc1.value()
        assert merged.value() >= gc2.value()

    def test_serialization_roundtrip(self):
        gc = GCounter("r1", {"r1": 5, "r2": 3})
        data = gc.to_dict()
        restored = GCounter.from_dict(data)
        assert restored.value() == gc.value()


# ===========================================================================
# PN-Counter Tests
# ===========================================================================

class TestPNCounter:
    def test_increment_decrement(self):
        pnc = PNCounter("r1")
        pnc.increment(10)
        pnc.decrement(3)
        assert pnc.value() == 7

    def test_negative_increment_raises(self):
        pnc = PNCounter("r1")
        try:
            pnc.increment(-1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_commutativity(self):
        pnc1 = PNCounter("r1", positive={"r1": 5}, negative={"r1": 2})
        pnc2 = PNCounter("r2", positive={"r2": 3}, negative={"r2": 1})
        assert pnc1.merge(pnc2).value() == pnc2.merge(pnc1).value()

    def test_idempotency(self):
        pnc = PNCounter("r1", positive={"r1": 5}, negative={"r1": 2})
        assert pnc.merge(pnc).value() == pnc.value()

    def test_serialization_roundtrip(self):
        pnc = PNCounter("r1", positive={"r1": 10}, negative={"r1": 3})
        data = pnc.to_dict()
        restored = PNCounter.from_dict(data)
        assert restored.value() == pnc.value()


# ===========================================================================
# LWW-Register Tests
# ===========================================================================

class TestLWWRegister:
    def test_set_get(self):
        reg = LWWRegister("r1")
        reg.set("hello")
        assert reg.get() == "hello"

    def test_merge_keeps_latest(self):
        reg1 = LWWRegister("r1")
        reg2 = LWWRegister("r2")
        reg1.set("value1")
        time.sleep(0.001)  # Ensure different timestamps
        reg2.set("value2")
        merged = reg1.merge(reg2)
        assert merged.get() == "value2"  # reg2 is later

    def test_commutativity(self):
        reg1 = LWWRegister("r1")
        reg2 = LWWRegister("r2")
        reg1.set("a")
        time.sleep(0.001)
        reg2.set("b")
        # Both merges should pick the same winner
        m1 = reg1.merge(reg2)
        m2 = reg2.merge(reg1)
        assert m1.get() == m2.get()

    def test_idempotency(self):
        reg = LWWRegister("r1")
        reg.set("value")
        merged = reg.merge(reg)
        assert merged.get() == reg.get()

    def test_serialization_roundtrip(self):
        reg = LWWRegister("r1")
        reg.set("test_value")
        data = reg.to_dict()
        restored = LWWRegister.from_dict(data)
        assert restored.get() == reg.get()


# ===========================================================================
# OR-Set Tests
# ===========================================================================

class TestORSet:
    def test_add_contains(self):
        ors = ORSet("r1")
        ors.add("item1")
        assert "item1" in ors
        assert len(ors) == 1

    def test_remove(self):
        ors = ORSet("r1")
        ors.add("item1")
        ors.remove("item1")
        assert "item1" not in ors

    def test_concurrent_add_wins_over_remove(self):
        ors1 = ORSet("r1")
        ors2 = ORSet("r2")
        ors1.add("shared")
        ors2.add("shared")  # Same element, different tags
        ors1.remove("shared")  # r1 removes its observed tags
        # After merge, r2's tags should still be there
        merged = ors1.merge(ors2)
        assert "shared" in merged, "OR-Set add-wins: concurrent add should survive remove"

    def test_commutativity(self):
        ors1 = ORSet("r1")
        ors2 = ORSet("r2")
        ors1.add("a")
        ors2.add("b")
        m1 = ors1.merge(ors2)
        m2 = ors2.merge(ors1)
        assert m1.members() == m2.members()

    def test_associativity(self):
        ors1 = ORSet("r1"); ors1.add("a")
        ors2 = ORSet("r2"); ors2.add("b")
        ors3 = ORSet("r3"); ors3.add("c")
        left = ors1.merge(ors2).merge(ors3)
        right = ors1.merge(ors2.merge(ors3))
        assert left.members() == right.members()

    def test_idempotency(self):
        ors = ORSet("r1")
        ors.add("item")
        merged = ors.merge(ors)
        assert merged.members() == ors.members()

    def test_serialization_roundtrip(self):
        ors = ORSet("r1")
        ors.add("x"); ors.add("y"); ors.add("z")
        data = ors.to_dict()
        restored = ORSet.from_dict(data)
        assert restored.members() == ors.members()
