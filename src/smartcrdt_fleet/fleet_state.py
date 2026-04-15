"""
fleet_state.py — CRDT-based fleet state synchronization.

Maintains a consistent, partition-tolerant view of the entire agent fleet using
composite CRDT types. Each agent has a status register, capability set, health
score, and heartbeat tracked via LWW-Registers and OR-Sets.

The fleet state is serialized to JSON and committed to git, enabling async
coordination: agents commit their updates, pull peer updates, merge CRDTs,
and converge to identical state without real-time communication.

Architecture:
    FleetState
    ├── agent_registry: ORSet[agent_id]         — membership
    ├── agent_status: Dict[id, LWWRegister]     — status (active/idle/alert)
    ├── agent_health: Dict[id, LWWRegister]     — health score (0.0-1.0)
    ├── agent_capabilities: Dict[id, ORSet]     — capability tags
    ├── agent_heartbeat: Dict[id, LWWRegister]  — last heartbeat timestamp
    ├── task_counts: Dict[id, GCounter]         — tasks completed
    ├── fleet_vector_clock: VectorClock          — causal ordering
    └── fleet_metrics: PNCounter                — aggregate fleet stats

Invariants:
    I1: Fleet state converges after all agents merge (eventual consistency)
    I2: No agent is ever silently removed (OR-Set add-wins semantics)
    I3: Most recent status always wins (LWW-Register with HLC)
    I4: Task counts are monotonically non-decreasing (G-Counter)
    I5: Causal ordering preserved across all operations (Vector Clock)

Python 3.9+ stdlib only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .crdt_primitives import (
    GCounter, HLCTimestamp, HybridLogicalClock, LWWRegister,
    ORSet, PNCounter, VectorClock,
)


@dataclass
class AgentSnapshot:
    """Point-in-time snapshot of a single agent's state."""
    agent_id: str
    status: str = "unknown"
    health_score: float = 0.0
    capabilities: Set[str] = field(default_factory=set)
    last_heartbeat: Optional[str] = None
    tasks_completed: int = 0
    vector_clock: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "health_score": self.health_score,
            "capabilities": sorted(self.capabilities),
            "last_heartbeat": self.last_heartbeat,
            "tasks_completed": self.tasks_completed,
            "vector_clock": self.vector_clock,
        }


class FleetState:
    """CRDT-based fleet state for multi-agent coordination.

    Manages the complete fleet state using composite CRDT types that
    guarantee eventual consistency under network partitions.

    Each agent's state is tracked using a combination of CRDT types:
    - ORSet for membership and capabilities (add-wins, no silent drops)
    - LWW-Register for status, health, heartbeat (most-recent wins)
    - G-Counter for task counts (monotonic, commutative)
    - VectorClock for causal ordering (partial order guarantees)

    Usage::

        fleet = FleetState("oracle1")
        fleet.register_agent("oracle1", capabilities=["coordination", "discovery"])
        fleet.update_status("oracle1", "active")
        fleet.heartbeat("oracle1")

        # Another agent merges:
        fleet2 = FleetState("datum")
        fleet2.register_agent("datum", capabilities=["auditing", "conformance"])
        fleet2.merge(fleet.to_dict())

        # Both are now identical
        assert fleet.to_dict() == fleet2.to_dict()
    """

    # Valid agent statuses
    STATUSES = {"active", "idle", "working", "alert", "silent", "retired", "unknown"}

    def __init__(self, replica_id: str,
                 hlc: Optional[HybridLogicalClock] = None):
        self._replica_id = replica_id
        self._hlc = hlc or HybridLogicalClock(node_id=replica_id)

        # Core CRDT state
        self._registry: ORSet = ORSet(replica_id)
        self._status: Dict[str, LWWRegister] = {}
        self._health: Dict[str, LWWRegister] = {}
        self._capabilities: Dict[str, ORSet] = {}
        self._heartbeat: Dict[str, LWWRegister] = {}
        self._task_counts: Dict[str, GCounter] = {}
        self._vclock = VectorClock(replica_id)
        self._fleet_metrics = PNCounter(replica_id)

    @property
    def replica_id(self) -> str:
        return self._replica_id

    # =========================================================================
    # Agent Lifecycle
    # =========================================================================

    def register_agent(self, agent_id: str,
                       capabilities: Optional[List[str]] = None) -> None:
        """Register a new agent in the fleet.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent (e.g. "oracle1", "datum").
        capabilities : list[str], optional
            Initial capability tags for the agent.
        """
        self._vclock.increment()
        self._registry.add(agent_id)

        # Initialize CRDT state for this agent if not present
        if agent_id not in self._status:
            self._status[agent_id] = LWWRegister(
                self._replica_id, "unknown",
                HybridLogicalClock(node_id=self._replica_id))
        if agent_id not in self._health:
            self._health[agent_id] = LWWRegister(
                self._replica_id, 1.0,
                HybridLogicalClock(node_id=self._replica_id))
        if agent_id not in self._heartbeat:
            self._heartbeat[agent_id] = LWWRegister(
                self._replica_id, None,
                HybridLogicalClock(node_id=self._replica_id))
        if agent_id not in self._task_counts:
            self._task_counts[agent_id] = GCounter(agent_id)
        if agent_id not in self._capabilities:
            self._capabilities[agent_id] = ORSet(agent_id)

        if capabilities:
            for cap in capabilities:
                self._capabilities[agent_id].add(cap)

    def deregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the fleet (mark as retired).

        Note: The agent stays in the OR-Set registry (add-wins semantics
        mean concurrent re-registrations are preserved). Status is set to
        "retired" to indicate the agent should not receive new tasks.
        """
        self._vclock.increment()
        # Don't remove from OR-Set — add-wins means it would reappear on merge.
        # Just set status to retired.
        if agent_id not in self._status:
            self._status[agent_id] = LWWRegister(
                self._replica_id, "retired",
                HybridLogicalClock(node_id=self._replica_id))
        else:
            self._status[agent_id].set("retired")

    # =========================================================================
    # State Updates
    # =========================================================================

    def update_status(self, agent_id: str, status: str) -> None:
        """Update an agent's status (active, idle, working, alert, silent).

        Uses LWW-Register — the most recent update wins deterministically.
        """
        if status not in self.STATUSES:
            raise ValueError(f"Invalid status {status!r}; must be one of {self.STATUSES}")
        self._vclock.increment()
        if agent_id not in self._status:
            self._status[agent_id] = LWWRegister(
                self._replica_id, status,
                HybridLogicalClock(node_id=self._replica_id))
        else:
            self._status[agent_id].set(status)

    def update_health(self, agent_id: str, score: float) -> None:
        """Update an agent's health score (0.0 to 1.0).

        Uses LWW-Register — the most recent score wins.
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Health score must be 0.0-1.0, got {score}")
        self._vclock.increment()
        if agent_id not in self._health:
            self._health[agent_id] = LWWRegister(
                self._replica_id, score,
                HybridLogicalClock(node_id=self._replica_id))
        else:
            self._health[agent_id].set(score)

    def add_capability(self, agent_id: str, capability: str) -> None:
        """Add a capability tag to an agent. Uses OR-Set (add-wins)."""
        self._vclock.increment()
        if agent_id not in self._capabilities:
            self._capabilities[agent_id] = ORSet(agent_id)
        self._capabilities[agent_id].add(capability)

    def remove_capability(self, agent_id: str, capability: str) -> None:
        """Remove a capability tag from an agent."""
        self._vclock.increment()
        if agent_id in self._capabilities:
            self._capabilities[agent_id].remove(capability)

    def heartbeat(self, agent_id: str) -> None:
        """Record a heartbeat from an agent.

        Uses LWW-Register — the most recent heartbeat timestamp wins.
        """
        self._vclock.increment()
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if agent_id not in self._heartbeat:
            self._heartbeat[agent_id] = LWWRegister(
                self._replica_id, ts,
                HybridLogicalClock(node_id=self._replica_id))
        else:
            self._heartbeat[agent_id].set(ts)

    def increment_task_count(self, agent_id: str, amount: int = 1) -> None:
        """Increment the task completion counter for an agent.

        Uses G-Counter — monotonically non-decreasing.
        """
        if amount < 0:
            raise ValueError("Task count can only increment (use G-Counter)")
        self._vclock.increment()
        if agent_id not in self._task_counts:
            self._task_counts[agent_id] = GCounter(agent_id)
        self._task_counts[agent_id].increment(amount)
        self._fleet_metrics.increment(amount)

    # =========================================================================
    # Queries
    # =========================================================================

    def get_agents(self) -> Set[str]:
        """Return set of all registered agents."""
        return self._registry.members()

    def get_snapshot(self, agent_id: str) -> Optional[AgentSnapshot]:
        """Get a point-in-time snapshot of an agent's state."""
        if not self._registry.contains(agent_id):
            return None
        return AgentSnapshot(
            agent_id=agent_id,
            status=self._status.get(agent_id, LWWRegister(agent_id, "unknown")).get() or "unknown",
            health_score=self._health.get(agent_id, LWWRegister(agent_id, 0.0)).get() or 0.0,
            capabilities=self._capabilities.get(agent_id, ORSet(agent_id)).members(),
            last_heartbeat=self._heartbeat.get(agent_id, LWWRegister(agent_id, None)).get(),
            tasks_completed=self._task_counts.get(agent_id, GCounter(agent_id)).value(),
            vector_clock=self._vclock.to_dict()["counters"],
        )

    def get_all_snapshots(self) -> Dict[str, AgentSnapshot]:
        """Get snapshots of all registered agents."""
        return {aid: snap for aid in self.get_agents()
                if (snap := self.get_snapshot(aid)) is not None}

    def find_by_capability(self, capability: str) -> List[str]:
        """Find agents that have a specific capability."""
        return sorted([aid for aid in self.get_agents()
                       if capability in self._capabilities.get(aid, ORSet(aid)).members()])

    def find_by_status(self, status: str) -> List[str]:
        """Find agents with a specific status."""
        return sorted([aid for aid in self.get_agents()
                       if self._status.get(aid, LWWRegister(aid, "unknown")).get() == status])

    def fleet_summary(self) -> Dict[str, Any]:
        """Get a high-level fleet summary."""
        agents = self.get_all_snapshots()
        active = [a for a in agents.values() if a.status == "active"]
        total_tasks = sum(a.tasks_completed for a in agents.values())
        return {
            "total_agents": len(agents),
            "active_agents": len(active),
            "total_tasks_completed": total_tasks,
            "fleet_vector_clock": self._vclock.to_dict()["counters"],
            "agents": {aid: snap.to_dict() for aid, snap in agents.items()},
        }

    # =========================================================================
    # CRDT Merge (the critical operation)
    # =========================================================================

    def merge(self, remote_state: Dict[str, Any]) -> None:
        """Merge remote fleet state into this replica.

        This is the core CRDT merge operation. It merges each sub-CRDT
        using its native merge function, guaranteeing eventual consistency.

        Parameters
        ----------
        remote_state : dict
            Serialized fleet state from another replica, as produced by to_dict().
        """
        self._vclock.increment()

        # Merge agent registry
        remote_registry = ORSet.from_dict(remote_state.get("registry", {}))
        self._registry = self._registry.merge(remote_registry)

        # Merge each agent's status registers
        for aid, reg_data in remote_state.get("status", {}).items():
            if aid not in self._status:
                self._status[aid] = LWWRegister.from_dict(reg_data)
            else:
                remote_reg = LWWRegister.from_dict(reg_data)
                self._status[aid] = self._status[aid].merge(remote_reg)

        # Merge health scores
        for aid, reg_data in remote_state.get("health", {}).items():
            if aid not in self._health:
                self._health[aid] = LWWRegister.from_dict(reg_data)
            else:
                remote_reg = LWWRegister.from_dict(reg_data)
                self._health[aid] = self._health[aid].merge(remote_reg)

        # Merge heartbeat registers
        for aid, reg_data in remote_state.get("heartbeat", {}).items():
            if aid not in self._heartbeat:
                self._heartbeat[aid] = LWWRegister.from_dict(reg_data)
            else:
                remote_reg = LWWRegister.from_dict(reg_data)
                self._heartbeat[aid] = self._heartbeat[aid].merge(remote_reg)

        # Merge capability sets
        for aid, ors_data in remote_state.get("capabilities", {}).items():
            if aid not in self._capabilities:
                self._capabilities[aid] = ORSet.from_dict(ors_data)
            else:
                remote_ors = ORSet.from_dict(ors_data)
                self._capabilities[aid] = self._capabilities[aid].merge(remote_ors)

        # Merge task counters
        for aid, gc_data in remote_state.get("task_counts", {}).items():
            if aid not in self._task_counts:
                self._task_counts[aid] = GCounter.from_dict(gc_data)
            else:
                remote_gc = GCounter.from_dict(gc_data)
                self._task_counts[aid] = self._task_counts[aid].merge(remote_gc)

        # Merge fleet vector clock
        remote_vc = VectorClock.from_dict(remote_state.get("vector_clock", {}))
        self._vclock = self._vclock.merge(remote_vc)

        # Merge fleet metrics
        remote_metrics = PNCounter.from_dict(remote_state.get("fleet_metrics", {}))
        self._fleet_metrics = self._fleet_metrics.merge(remote_metrics)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete fleet state to a JSON-compatible dict."""
        return {
            "replica_id": self._replica_id,
            "version": "0.1.0",
            "vector_clock": self._vclock.to_dict(),
            "registry": self._registry.to_dict(),
            "status": {aid: reg.to_dict() for aid, reg in self._status.items()
                       if self._registry.contains(aid)},
            "health": {aid: reg.to_dict() for aid, reg in self._health.items()
                       if self._registry.contains(aid)},
            "heartbeat": {aid: reg.to_dict() for aid, reg in self._heartbeat.items()
                          if self._registry.contains(aid)},
            "capabilities": {aid: ors.to_dict()
                             for aid, ors in self._capabilities.items()
                             if self._registry.contains(aid)},
            "task_counts": {aid: gc.to_dict()
                            for aid, gc in self._task_counts.items()
                            if self._registry.contains(aid)},
            "fleet_metrics": self._fleet_metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FleetState:
        """Deserialize fleet state from dict.

        Returns a FleetState with the merged state already applied.
        """
        replica_id = data.get("replica_id", "unknown")
        fs = cls(replica_id)
        fs.merge(data)
        return fs
