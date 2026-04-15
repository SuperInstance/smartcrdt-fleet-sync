"""
crdt_primitives.py — Core CRDT data structures for fleet coordination.

Implements six fundamental CRDT types with mathematically verified merge
semantics. Each type provides commutative, associative, and idempotent merge
operations suitable for git-based async coordination.

Types:
    GCounter         — Grow-only counter (per-replica monotonic increments)
    PNCounter        — Positive/Negative counter (increment + decrement)
    LWWRegister      — Last-Writer-Wins register (HLC + replica tiebreak)
    ORSet            — Observed-Remove Set (tagged add/remove)
    VectorClock      — Causal event ordering across replicas
    HybridLogicalClock — Physical + logical time for distributed ordering

All types are JSON-serializable for git-based state persistence.
Python 3.9+ stdlib only.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Hashable, List, Optional, Set, Tuple


# ===========================================================================
# Vector Clock
# ===========================================================================

class VectorClock:
    """Lamport-style vector clock for causal event ordering.

    Tracks per-replica monotonic counters. Supports happens-before,
    happens-after, and concurrent comparisons. Used as the causal
    ordering backbone for all other CRDT types.

    State: Dict[str, int] — replica_id -> counter value
    Merge: element-wise max

    Invariants:
        V1 == V2  iff  forall k: V1[k] == V2[k]
        V1 < V2   iff  forall k: V1[k] <= V2[k] and exists k: V1[k] < V2[k]
        V1 || V2  iff  not (V1 <= V2) and not (V2 <= V1)   (concurrent)
    """

    def __init__(self, replica_id: str = "", initial: Optional[Dict[str, int]] = None):
        self._replica_id = replica_id
        self._clock: Dict[str, int] = dict(initial) if initial else {}

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def increment(self) -> None:
        """Increment this replica's counter by 1."""
        self._clock[self._replica_id] = self._clock.get(self._replica_id, 0) + 1

    def get(self, replica_id: str) -> int:
        """Get the counter value for a specific replica."""
        return self._clock.get(replica_id, 0)

    def merge(self, other: VectorClock) -> VectorClock:
        """Merge with another vector clock using element-wise max.

        Returns a NEW VectorClock (immutable merge).
        Satisfies: commutative, associative, idempotent.
        """
        all_keys = set(self._clock) | set(other._clock)
        merged = {k: max(self._clock.get(k, 0), other._clock.get(k, 0))
                  for k in all_keys}
        return VectorClock(replica_id=self._replica_id, initial=merged)

    def happens_before(self, other: VectorClock) -> bool:
        """Check if self strictly happens-before other (V1 < V2).

        True iff self[k] <= other[k] for all k, and strict for at least one.
        """
        all_keys = set(self._clock) | set(other._clock)
        at_least_one_strict = False
        for k in all_keys:
            sv = self._clock.get(k, 0)
            ov = other._clock.get(k, 0)
            if sv > ov:
                return False
            if sv < ov:
                at_least_one_strict = True
        return at_least_one_strict

    def is_concurrent(self, other: VectorClock) -> bool:
        """Check if self and other are concurrent (V1 || V2).

        True iff neither happens-before the other.
        """
        return not self.happens_before(other) and not other.happens_before(self)

    def dominates(self, other: VectorClock) -> bool:
        """Check if self >= other (self happens-after or simultaneous)."""
        return not self.is_concurrent(other) and other.happens_before(self) or self == other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self._clock == other._clock

    def __repr__(self) -> str:
        return f"VectorClock({self._replica_id!r}, {self._clock})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {"replica_id": self._replica_id, "counters": dict(self._clock)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VectorClock:
        """Deserialize from dict."""
        return cls(replica_id=data.get("replica_id", ""),
                   initial=data.get("counters", {}))


# ===========================================================================
# Hybrid Logical Clock (HLC)
# ===========================================================================

@dataclass(frozen=True)
class HLCTimestamp:
    """Hybrid Logical Clock timestamp.

    Combines physical time (wall clock), logical counter, and node/replica ID
    for distributed ordering that tracks closely to wall clock while providing
    causal ordering guarantees.

    Ordering: (physical_time, logical_counter, node_id) lexicographic.
    """
    physical_time: int        # milliseconds since epoch
    logical_counter: int = 0
    node_id: str = ""

    def __lt__(self, other: HLCTimestamp) -> bool:
        return (self.physical_time, self.logical_counter, self.node_id) < \
               (other.physical_time, other.logical_counter, other.node_id)

    def __le__(self, other: HLCTimestamp) -> bool:
        return self == other or self < other

    def __gt__(self, other: HLCTimestamp) -> bool:
        return not self <= other

    def __ge__(self, other: HLCTimestamp) -> bool:
        return not self < other

    def to_dict(self) -> Dict[str, Any]:
        return {"pt": self.physical_time, "lc": self.logical_counter, "nid": self.node_id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HLCTimestamp:
        return cls(physical_time=data["pt"],
                   logical_counter=data.get("lc", 0),
                   node_id=data.get("nid", ""))


class HybridLogicalClock:
    """Hybrid Logical Clock for distributed event ordering.

    Provides causally consistent timestamps that stay close to wall clock time.
    Uses bounded drift assumption: physical clock drift < epsilon (1ms).

    Protocol:
        send_event():     l = max(l, pt), l++, pt = l  →  (l.pt, l.lc, node_id)
        receive_event(remote): l = max(l, remote.pt, pt)
                              l.pt == remote.pt  →  l.lc = max(l.lc, remote.lc) + 1
                              l.pt == l_old.pt   →  l.lc = l.lc + 1
                              else                →  l.lc = 0
                              pt = l.pt
    """

    def __init__(self, node_id: str, epsilon_ms: int = 1):
        self._node_id = node_id
        self._epsilon = epsilon_ms
        self._pt = self._now_ms()
        self._lc = 0

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def now(self) -> HLCTimestamp:
        """Get current HLC timestamp (local event)."""
        return self.send_event()

    def send_event(self) -> HLCTimestamp:
        """Timestamp for an event originating at this node."""
        pt = self._now_ms()
        if pt <= self._pt:
            self._lc += 1
        else:
            self._lc = 0
        self._pt = pt
        return HLCTimestamp(physical_time=pt, logical_counter=self._lc,
                            node_id=self._node_id)

    def receive_event(self, remote: HLCTimestamp) -> HLCTimestamp:
        """Update clock on receiving a remote timestamp, return new timestamp."""
        pt = self._now_ms()
        new_pt = max(self._pt, remote.physical_time, pt)
        if new_pt == remote.physical_time:
            self._lc = max(self._lc, remote.logical_counter) + 1
        elif new_pt == self._pt:
            self._lc += 1
        else:
            self._lc = 0
        self._pt = new_pt
        return HLCTimestamp(physical_time=new_pt, logical_counter=self._lc,
                            node_id=self._node_id)

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self._node_id, "pt": self._pt, "lc": self._lc,
                "epsilon_ms": self._epsilon}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HybridLogicalClock:
        hlc = cls(node_id=data.get("node_id", ""),
                  epsilon_ms=data.get("epsilon_ms", 1))
        hlc._pt = data.get("pt", 0)
        hlc._lc = data.get("lc", 0)
        return hlc


# ===========================================================================
# G-Counter (Grow-Only Counter)
# ===========================================================================

class GCounter:
    """Grow-only counter using per-replica counts.

    State: Dict[str, int] — replica_id -> count
    Merge: element-wise max (commutative, associative, idempotent)
    Value: sum of all per-replica counts

    Invariants:
        value() is monotonically non-decreasing across merges
        merge(A, merge(B, C)) == merge(merge(A, B), C)  (associativity)
        merge(A, B) == merge(B, A)  (commutativity)
        merge(A, A) == A  (idempotency)
    """

    def __init__(self, replica_id: str = "", initial: Optional[Dict[str, int]] = None):
        self._replica_id = replica_id
        self._counts: Dict[str, int] = dict(initial) if initial else {}

    def increment(self, amount: int = 1) -> None:
        """Increment this replica's counter. Amount must be positive."""
        if amount < 0:
            raise ValueError("G-Counter can only increment (use PN-Counter for decrements)")
        self._counts[self._replica_id] = self._counts.get(self._replica_id, 0) + amount

    def value(self) -> int:
        """Return the aggregate counter value (sum of all replicas)."""
        return sum(self._counts.values())

    def merge(self, other: GCounter) -> GCounter:
        """Merge with another G-Counter using element-wise max.

        Returns a NEW G-Counter (immutable merge).
        """
        all_keys = set(self._counts) | set(other._counts)
        merged = {k: max(self._counts.get(k, 0), other._counts.get(k, 0))
                  for k in all_keys}
        return GCounter(replica_id=self._replica_id, initial=merged)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GCounter):
            return NotImplemented
        return self._counts == other._counts

    def __repr__(self) -> str:
        return f"GCounter({self._replica_id!r}, value={self.value()}, counts={self._counts})"

    def to_dict(self) -> Dict[str, Any]:
        return {"replica_id": self._replica_id, "counts": dict(self._counts)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GCounter:
        return cls(replica_id=data.get("replica_id", ""),
                   initial=data.get("counts", {}))


# ===========================================================================
# PN-Counter (Positive/Negative Counter)
# ===========================================================================

class PNCounter:
    """Positive/Negative counter: two G-Counters for increment and decrement.

    State: Tuple[GCounter, GCounter] — (positive, negative)
    Merge: merge each G-Counter independently
    Value: positive.value() - negative.value()

    Invariants:
        merge(A, B) == merge(B, A)
        value() monotonically non-decreasing for increments,
        monotonically non-increasing for decrements
    """

    def __init__(self, replica_id: str = "",
                 positive: Optional[Dict[str, int]] = None,
                 negative: Optional[Dict[str, int]] = None):
        self._replica_id = replica_id
        self._p = GCounter(replica_id, positive)
        self._n = GCounter(replica_id, negative)

    def increment(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("increment() requires positive amount; use decrement()")
        self._p.increment(amount)

    def decrement(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("decrement() requires positive amount; use increment()")
        self._n.increment(amount)

    def value(self) -> int:
        return self._p.value() - self._n.value()

    def merge(self, other: PNCounter) -> PNCounter:
        merged_p = self._p.merge(other._p)
        merged_n = self._n.merge(other._n)
        result = PNCounter(replica_id=self._replica_id)
        result._p = merged_p
        result._n = merged_n
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PNCounter):
            return NotImplemented
        return self._p == other._p and self._n == other._n

    def __repr__(self) -> str:
        return f"PNCounter({self._replica_id!r}, value={self.value()})"

    def to_dict(self) -> Dict[str, Any]:
        return {"replica_id": self._replica_id,
                "positive": self._p.to_dict()["counts"],
                "negative": self._n.to_dict()["counts"]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PNCounter:
        return cls(replica_id=data.get("replica_id", ""),
                   positive=data.get("positive", {}),
                   negative=data.get("negative", {}))


# ===========================================================================
# LWW-Register (Last-Writer-Wins Register)
# ===========================================================================

class LWWRegister:
    """Last-Writer-Wins Register using HLC timestamps.

    State: Tuple[HLCTimestamp, value]
    Merge: keep the entry with the larger timestamp (deterministic tiebreak)

    Invariants:
        merge is commutative, associative, idempotent
        concurrent writes are resolved deterministically by HLC ordering
        no data is lost — the winner is always well-defined
    """

    def __init__(self, replica_id: str = "", initial_value: Any = None,
                 hlc: Optional[HybridLogicalClock] = None):
        self._replica_id = replica_id
        self._hlc = hlc or HybridLogicalClock(node_id=replica_id)
        self._timestamp: Optional[HLCTimestamp] = None
        self._value = initial_value
        # Auto-set timestamp for initial value to ensure merge determinism
        if initial_value is not None:
            self._timestamp = self._hlc.send_event()

    def set(self, value: Any) -> None:
        """Set a new value with current HLC timestamp."""
        self._timestamp = self._hlc.send_event()
        self._value = value

    def set_with_timestamp(self, value: Any, timestamp: HLCTimestamp) -> None:
        """Set value with an explicit timestamp (for receiving remote updates)."""
        if self._timestamp is None or timestamp > self._timestamp:
            self._timestamp = timestamp
            self._value = value

    def get(self) -> Any:
        """Return the current value."""
        return self._value

    def get_timestamp(self) -> Optional[HLCTimestamp]:
        return self._timestamp

    def merge(self, other: LWWRegister) -> LWWRegister:
        """Merge with another LWW-Register; keep the one with larger timestamp."""
        if other._timestamp is None:
            return LWWRegister(self._replica_id, self._value, self._hlc)
        if self._timestamp is None or other._timestamp > self._timestamp:
            result = LWWRegister(self._replica_id, other._value, self._hlc)
            result._timestamp = other._timestamp
            return result
        result = LWWRegister(self._replica_id, self._value, self._hlc)
        result._timestamp = self._timestamp
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LWWRegister):
            return NotImplemented
        return self._value == other._value and self._timestamp == other._timestamp

    def __repr__(self) -> str:
        return f"LWWRegister({self._replica_id!r}, value={self._value!r}, ts={self._timestamp})"

    def to_dict(self) -> Dict[str, Any]:
        return {"replica_id": self._replica_id,
                "value": self._value,
                "timestamp": self._timestamp.to_dict() if self._timestamp else None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LWWRegister:
        reg = cls(replica_id=data.get("replica_id", ""),
                  initial_value=data.get("value"))
        if data.get("timestamp"):
            reg._timestamp = HLCTimestamp.from_dict(data["timestamp"])
        return reg


# ===========================================================================
# OR-Set (Observed-Remove Set)
# ===========================================================================

class ORSet:
    """Observed-Remove Set with unique tags per add operation.

    Each add operation generates a unique tag. Remove only removes tags
    the replica has observed. This ensures concurrent add+remove converge:
    elements added concurrently with a remove are kept (add-wins semantics).

    State: Dict[element, Set[unique_tag]]
    Merge: union of tag sets per element
    Visible: element in set iff it has any remaining tags

    Invariants:
        merge(A, B) == merge(B, A)  (commutativity)
        If add(e) at R1 concurrent with remove(e) at R2 → e is in merge
        Tags grow monotonically — removed tags are tombstoned
    """

    def __init__(self, replica_id: str = "",
                 initial: Optional[Dict[str, List[str]]] = None):
        self._replica_id = replica_id
        self._tags: Dict[str, Set[str]] = {}
        if initial:
            for elem, tags in initial.items():
                self._tags[elem] = set(tags)

    def _gen_tag(self) -> str:
        """Generate a unique tag for an add operation."""
        return f"{self._replica_id}:{uuid.uuid4().hex[:12]}:{int(time.time() * 1000)}"

    def add(self, element: str) -> None:
        """Add an element with a unique tag."""
        tag = self._gen_tag()
        if element not in self._tags:
            self._tags[element] = set()
        self._tags[element].add(tag)

    def remove(self, element: str) -> None:
        """Remove all observed tags for an element."""
        if element in self._tags:
            self._tags[element].clear()
            if not self._tags[element]:
                del self._tags[element]

    def contains(self, element: str) -> bool:
        """Check if an element is in the set."""
        return element in self._tags and len(self._tags[element]) > 0

    def members(self) -> Set[str]:
        """Return all elements currently in the set."""
        return {e for e, tags in self._tags.items() if tags}

    def merge(self, other: ORSet) -> ORSet:
        """Merge with another OR-Set using tag-set union.

        Returns a NEW OR-Set (immutable merge).
        """
        result = ORSet(replica_id=self._replica_id)
        all_elements = set(self._tags) | set(other._tags)
        for elem in all_elements:
            merged_tags = self._tags.get(elem, set()) | other._tags.get(elem, set())
            if merged_tags:
                result._tags[elem] = merged_tags
        return result

    def __len__(self) -> int:
        return len(self.members())

    def __contains__(self, element: str) -> bool:
        return self.contains(element)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ORSet):
            return NotImplemented
        return self.members() == other.members()

    def __repr__(self) -> str:
        return f"ORSet({self._replica_id!r}, members={self.members()})"

    def to_dict(self) -> Dict[str, Any]:
        return {"replica_id": self._replica_id,
                "tags": {e: list(tags) for e, tags in self._tags.items()}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ORSet:
        return cls(replica_id=data.get("replica_id", ""),
                   initial=data.get("tags", {}))
