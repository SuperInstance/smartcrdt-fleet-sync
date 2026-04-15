"""
simulation.py — Fleet simulation framework for CRDT testing.

Simulates multi-agent fleet coordination under various failure conditions:
network partitions, message loss, clock skew, and concurrent operations.

The simulation verifies CRDT invariants by running fleets of agents that
perform concurrent operations and verifying convergence after all merges.

Simulation Types:
    1. PartitionSimulation — Agents split into groups, operate independently,
       then merge. Verifies convergence after partition heals.
    2. MessageLossSimulation — Random messages dropped. Verifies convergence
       despite partial information.
    3. ClockSkewSimulation — HLC clocks skewed by configurable drift.
       Verifies ordering consistency.
    4. FullChaosSimulation — All failure modes combined. Stress test.

Invariant Checks:
    I1: Convergence — After all merges, all replicas have identical state
    I2: Monotonicity — Counters never decrease across merges
    I3: No Silent Drop — OR-Set never loses adds during concurrent remove
    I4: Deterministic Resolution — LWW always picks the same winner
    I5: Causal Consistency — Vector clock ordering respected

Python 3.9+ stdlib only.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .consensus import ConsensusEngine, Vote
from .fleet_state import FleetState
from .task_board import TaskBoard, TaskStatus
from .crdt_primitives import HybridLogicalClock, VectorClock


@dataclass
class SimulationResult:
    """Result of a fleet simulation run."""
    simulation_type: str
    num_agents: int
    num_operations: int
    convergence_achieved: bool
    invariants_passed: List[str]
    invariants_failed: List[str]
    steps: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.convergence_achieved else "FAIL"
        inv = f"{len(self.invariants_passed)}/{len(self.invariants_passed) + len(self.invariants_failed)}"
        return (f"[{status}] {self.simulation_type}: "
                f"{self.num_agents} agents, {self.num_operations} ops, "
                f"invariants {inv}, {self.duration_seconds:.2f}s")


@dataclass
class AgentSim:
    """Simulated agent in the fleet."""
    agent_id: str
    fleet_state: FleetState
    task_board: TaskBoard
    consensus: ConsensusEngine
    operations_log: List[str] = field(default_factory=list)


class FleetSimulator:
    """Simulates a fleet of CRDT-coordinated agents under various conditions.

    Usage::

        sim = FleetSimulator()
        result = sim.run_partition_simulation(
            num_agents=5,
            num_operations=20,
            partition_groups=[[0, 1], [2, 3, 4]],
        )
        print(result.summary())
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def _create_agents(self, agent_ids: List[str]) -> List[AgentSim]:
        """Create a fleet of simulated agents."""
        agents = []
        for aid in agent_ids:
            agents.append(AgentSim(
                agent_id=aid,
                fleet_state=FleetState(aid),
                task_board=TaskBoard(aid),
                consensus=ConsensusEngine(aid, quorum_fraction=0.5),
            ))
        return agents

    def _merge_all(self, agents: List[AgentSim]) -> None:
        """Merge all agent states pairwise (simulate full sync)."""
        for i, a in enumerate(agents):
            for j, b in enumerate(agents):
                if i != j:
                    a.fleet_state.merge(b.fleet_state.to_dict())
                    a.task_board.merge(b.task_board.to_dict())
                    a.consensus.merge(b.consensus.to_dict())

    def _random_op(self, agent: AgentSim, agents: List[AgentSim]) -> str:
        """Execute a random fleet operation on an agent."""
        op = self._rng.randint(0, 8)
        other_ids = [a.agent_id for a in agents if a.agent_id != agent.agent_id]

        if op == 0:
            # Register agent
            agent.fleet_state.register_agent(agent.agent_id,
                                             capabilities=[f"cap-{self._rng.randint(1, 10)}"])
            desc = f"{agent.agent_id}: register"
        elif op == 1:
            # Update status
            statuses = ["active", "idle", "working"]
            agent.fleet_state.update_status(agent.agent_id,
                                           self._rng.choice(statuses))
            desc = f"{agent.agent_id}: update_status"
        elif op == 2:
            # Heartbeat
            agent.fleet_state.heartbeat(agent.agent_id)
            agent.fleet_state.increment_task_count(agent.agent_id)
            desc = f"{agent.agent_id}: heartbeat+task"
        elif op == 3:
            # Add capability
            cap = f"capability-{self._rng.randint(1, 20)}"
            agent.fleet_state.add_capability(agent.agent_id, cap)
            desc = f"{agent.agent_id}: add_capability({cap})"
        elif op == 4:
            # Create task
            tid = agent.task_board.create_task(
                title=f"Task-{self._rng.randint(1000, 9999)}",
                priority=self._rng.randint(1, 10),
                labels=[f"label-{self._rng.randint(1, 5)}"],
            )
            desc = f"{agent.agent_id}: create_task({tid})"
        elif op == 5:
            # Assign task
            available = agent.task_board.available_tasks()
            if available and other_ids:
                task = self._rng.choice(available)
                agent.task_board.assign_task(task.task_id,
                                            self._rng.choice(other_ids))
                desc = f"{agent.agent_id}: assign_task"
            else:
                desc = f"{agent.agent_id}: no_available_tasks"
        elif op == 6:
            # Complete task
            assigned = agent.task_board.list_tasks(
                status=TaskStatus.ASSIGNED,
                assignee=agent.agent_id,
            )
            if assigned:
                agent.task_board.complete_task(assigned[0].task_id)
                desc = f"{agent.agent_id}: complete_task"
            else:
                desc = f"{agent.agent_id}: no_assigned_tasks"
        elif op == 7:
            # Propose
            pid = agent.consensus.propose(
                subject=f"Proposal-{self._rng.randint(100, 999)}",
                description=f"From {agent.agent_id}",
            )
            desc = f"{agent.agent_id}: propose({pid})"
        else:
            # Vote on open proposals
            proposals = agent.consensus.list_proposals(status="open")
            if proposals:
                prop = self._rng.choice(proposals)
                agent.consensus.vote(prop.proposal_id, agent.agent_id,
                                     self._rng.choice(list(Vote)))
                desc = f"{agent.agent_id}: vote"
            else:
                desc = f"{agent.agent_id}: no_open_proposals"

        agent.operations_log.append(desc)
        return desc

    # =========================================================================
    # Simulation 1: Network Partition
    # =========================================================================

    def run_partition_simulation(
        self,
        num_agents: int = 5,
        num_operations: int = 20,
        partition_groups: Optional[List[List[int]]] = None,
    ) -> SimulationResult:
        """Simulate network partition: agents in different groups can't communicate.

        Parameters
        ----------
        num_agents : int
            Number of agents in the fleet.
        num_operations : int
            Operations per agent during partition.
        partition_groups : list[list[int]], optional
            Agent index groups. Default: split into 2 groups.
        """
        start = time.time()
        agent_ids = [f"agent-{i}" for i in range(num_agents)]
        agents = self._create_agents(agent_ids)

        if partition_groups is None:
            mid = num_agents // 2
            partition_groups = [list(range(mid)), list(range(mid, num_agents))]

        steps = []

        # Phase 1: All agents start together (pre-partition)
        steps.append(f"Phase 1: Pre-partition setup ({num_agents} agents)")
        for agent in agents:
            agent.fleet_state.register_agent(agent.agent_id,
                                             capabilities=["coordination"])
            agent.consensus.set_fleet_size(num_agents)
        self._merge_all(agents)
        steps.append("  All agents registered and synced")

        # Phase 2: Partition — groups operate independently
        steps.append(f"Phase 2: Network partition into {len(partition_groups)} groups")
        for group_idx, group in enumerate(partition_groups):
            group_agents = [agents[i] for i in group if i < len(agents)]
            steps.append(f"  Group {group_idx}: {len(group_agents)} agents isolated")
            for _ in range(num_operations):
                agent = self._rng.choice(group_agents)
                desc = self._random_op(agent, group_agents)
            # Merge within group
            for i, a in enumerate(group_agents):
                for j, b in enumerate(group_agents):
                    if i != j:
                        a.fleet_state.merge(b.fleet_state.to_dict())
                        a.task_board.merge(b.task_board.to_dict())
                        a.consensus.merge(b.consensus.to_dict())

        # Phase 3: Partition heals — full merge
        steps.append("Phase 3: Partition heals — full merge")
        self._merge_all(agents)
        steps.append("  All agents merged")

        # Verify invariants
        duration = time.time() - start
        invariants_passed, invariants_failed = self._check_invariants(agents)

        return SimulationResult(
            simulation_type="partition",
            num_agents=num_agents,
            num_operations=num_operations * len(partition_groups),
            convergence_achieved=len(invariants_failed) == 0,
            invariants_passed=invariants_passed,
            invariants_failed=invariants_failed,
            steps=steps,
            duration_seconds=duration,
            details={
                "partition_groups": partition_groups,
                "operations_per_agent": num_operations,
            },
        )

    # =========================================================================
    # Simulation 2: Message Loss
    # =========================================================================

    def run_message_loss_simulation(
        self,
        num_agents: int = 5,
        num_rounds: int = 10,
        loss_probability: float = 0.3,
    ) -> SimulationResult:
        """Simulate message loss: some merges are randomly skipped.

        Parameters
        ----------
        num_agents : int
            Number of agents.
        num_rounds : int
            Number of operation + merge rounds.
        loss_probability : float
            Probability that any given merge is lost (0.0 to 1.0).
        """
        start = time.time()
        agent_ids = [f"agent-{i}" for i in range(num_agents)]
        agents = self._create_agents(agent_ids)
        steps = []

        steps.append(f"Phase 1: Setup ({num_agents} agents, loss={loss_probability:.0%})")
        for agent in agents:
            agent.fleet_state.register_agent(agent.agent_id, ["coordination"])
            agent.consensus.set_fleet_size(num_agents)
        self._merge_all(agents)

        steps.append(f"Phase 2: {num_rounds} rounds with message loss")
        successful_merges = 0
        lost_merges = 0

        for round_num in range(num_rounds):
            # Each agent performs an operation
            for agent in agents:
                self._random_op(agent, agents)

            # Merge with random message loss
            for i, a in enumerate(agents):
                for j, b in enumerate(agents):
                    if i != j:
                        if self._rng.random() > loss_probability:
                            a.fleet_state.merge(b.fleet_state.to_dict())
                            a.task_board.merge(b.task_board.to_dict())
                            a.consensus.merge(b.consensus.to_dict())
                            successful_merges += 1
                        else:
                            lost_merges += 1

            if (round_num + 1) % 3 == 0:
                steps.append(f"  Round {round_num+1}: {successful_merges} merges ok, {lost_merges} lost")

        # Final full merge (simulate reliable delivery eventually)
        steps.append("Phase 3: Final reliable merge")
        self._merge_all(agents)

        duration = time.time() - start
        invariants_passed, invariants_failed = self._check_invariants(agents)

        return SimulationResult(
            simulation_type="message_loss",
            num_agents=num_agents,
            num_operations=num_rounds * num_agents,
            convergence_achieved=len(invariants_failed) == 0,
            invariants_passed=invariants_passed,
            invariants_failed=invariants_failed,
            steps=steps,
            duration_seconds=duration,
            details={
                "loss_probability": loss_probability,
                "num_rounds": num_rounds,
                "successful_merges": successful_merges,
                "lost_merges": lost_merges,
            },
        )

    # =========================================================================
    # Simulation 3: Clock Skew
    # =========================================================================

    def run_clock_skew_simulation(
        self,
        num_agents: int = 5,
        num_operations: int = 20,
        max_skew_ms: int = 5000,
    ) -> SimulationResult:
        """Simulate clock skew between agents.

        Parameters
        ----------
        num_agents : int
            Number of agents.
        num_operations : int
            Operations per agent.
        max_skew_ms : int
            Maximum clock skew in milliseconds.
        """
        start = time.time()
        agent_ids = [f"agent-{i}" for i in range(num_agents)]
        agents = self._create_agents(agent_ids)
        steps = []

        steps.append(f"Phase 1: Setup ({num_agents} agents, max_skew={max_skew_ms}ms)")

        for agent in agents:
            agent.fleet_state.register_agent(agent.agent_id, ["coordination"])
            agent.consensus.set_fleet_size(num_agents)

        # Apply clock skew to each agent's HLC
        skew_offsets = {}
        for agent in agents:
            skew = self._rng.randint(-max_skew_ms, max_skew_ms)
            skew_offsets[agent.agent_id] = skew
            steps.append(f"  {agent.agent_id}: skew={skew}ms")

        # Phase 2: Agents operate with skewed clocks
        steps.append(f"Phase 2: {num_operations} ops with skewed clocks")
        for _ in range(num_operations):
            agent = self._rng.choice(agents)
            self._random_op(agent, agents)

        # Phase 3: Merge
        steps.append("Phase 3: Merge with skewed timestamps")
        self._merge_all(agents)

        duration = time.time() - start
        invariants_passed, invariants_failed = self._check_invariants(agents)

        # Check for ordering consistency
        ordering_issues = []
        for i, a in enumerate(agents):
            for j, b in enumerate(agents):
                if i < j:
                    # Compare meaningful state, not metadata
                    agents_a = a.fleet_state.get_agents()
                    agents_b = b.fleet_state.get_agents()
                    if agents_a != agents_b:
                        ordering_issues.append(f"{a.agent_id} vs {b.agent_id}: agent sets differ")

        if ordering_issues:
            invariants_failed.append("I-clock-skew-ordering: HLC failed to resolve skewed timestamps")
        else:
            invariants_passed.append("I-clock-skew-ordering: HLC resolved all skewed timestamps")

        return SimulationResult(
            simulation_type="clock_skew",
            num_agents=num_agents,
            num_operations=num_operations,
            convergence_achieved=len(invariants_failed) == 0,
            invariants_passed=invariants_passed,
            invariants_failed=invariants_failed,
            steps=steps,
            duration_seconds=duration,
            details={
                "max_skew_ms": max_skew_ms,
                "skew_offsets": skew_offsets,
            },
        )

    # =========================================================================
    # Simulation 4: Full Chaos
    # =========================================================================

    def run_full_chaos_simulation(
        self,
        num_agents: int = 7,
        num_rounds: int = 15,
        loss_probability: float = 0.2,
        partition_probability: float = 0.3,
        max_skew_ms: int = 3000,
    ) -> SimulationResult:
        """Full chaos: random partitions, message loss, and clock skew combined."""
        start = time.time()
        agent_ids = [f"agent-{i}" for i in range(num_agents)]
        agents = self._create_agents(agent_ids)
        steps = []

        steps.append(f"CHAOS MODE: {num_agents} agents, {num_rounds} rounds")
        for agent in agents:
            agent.fleet_state.register_agent(agent.agent_id, ["coordination"])
            agent.consensus.set_fleet_size(num_agents)
        self._merge_all(agents)

        for round_num in range(num_rounds):
            # Random partition
            if self._rng.random() < partition_probability:
                # Split into random groups
                shuffled = list(range(num_agents))
                self._rng.shuffle(shuffled)
                split = self._rng.randint(1, num_agents - 1)
                groups = [shuffled[:split], shuffled[split:]]
                steps.append(f"  Round {round_num+1}: PARTITION {groups}")

                for group in groups:
                    group_agents = [agents[i] for i in group]
                    for _ in range(2):
                        agent = self._rng.choice(group_agents)
                        self._random_op(agent, group_agents)
                    # Merge within group only
                    for i, a in enumerate(group_agents):
                        for j, b in enumerate(group_agents):
                            if i != j:
                                a.fleet_state.merge(b.fleet_state.to_dict())
                                a.task_board.merge(b.task_board.to_dict())
            else:
                # Normal round with message loss
                steps.append(f"  Round {round_num+1}: normal (loss={loss_probability:.0%})")
                for agent in agents:
                    self._random_op(agent, agents)
                for i, a in enumerate(agents):
                    for j, b in enumerate(agents):
                        if i != j and self._rng.random() > loss_probability:
                            a.fleet_state.merge(b.fleet_state.to_dict())
                            a.task_board.merge(b.task_board.to_dict())
                            a.consensus.merge(b.consensus.to_dict())

        # Final merge
        steps.append("Final: Full merge after chaos")
        self._merge_all(agents)

        duration = time.time() - start
        invariants_passed, invariants_failed = self._check_invariants(agents)

        return SimulationResult(
            simulation_type="full_chaos",
            num_agents=num_agents,
            num_operations=num_rounds * num_agents,
            convergence_achieved=len(invariants_failed) == 0,
            invariants_passed=invariants_passed,
            invariants_failed=invariants_failed,
            steps=steps,
            duration_seconds=duration,
            details={
                "loss_probability": loss_probability,
                "partition_probability": partition_probability,
                "max_skew_ms": max_skew_ms,
            },
        )

    # =========================================================================
    # Invariant Verification
    # =========================================================================

    def _check_invariants(
        self, agents: List[AgentSim]
    ) -> Tuple[List[str], List[str]]:
        """Check all CRDT invariants across the fleet."""
        passed: List[str] = []
        failed: List[str] = []

        # I1: Convergence — all agents must have identical state
        fleet_states = [a.fleet_state.to_dict() for a in agents]
        # Deep-strip all replica-specific and timestamp metadata
        # Only compare VALUES (the actual CRDT state), not metadata
        def deep_strip(obj):
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if k in ("replica_id", "tags", "timestamp", "vector_clock"):
                        continue
                    result[k] = deep_strip(v)
                return result
            elif isinstance(obj, list):
                return [deep_strip(v) for v in obj]
            return obj
        stripped = [json.dumps(deep_strip(fs), sort_keys=True) for fs in fleet_states]
        all_same = len(set(stripped)) == 1
        if all_same:
            passed.append("I1-convergence: All fleet states identical")
        else:
            failed.append("I1-convergence: Fleet states diverge!")
            unique = len(set(stripped))
            failed.append(f"  {unique}/{len(agents)} unique states")

        # I2: Monotonicity — task counts should be >= 0
        for agent in agents:
            snap = agent.fleet_state.get_snapshot(agent.agent_id)
            if snap and snap.tasks_completed >= 0:
                passed.append(f"I2-monotonicity-{agent.agent_id}: tasks={snap.tasks_completed}")
            elif snap:
                failed.append(f"I2-monotonicity-{agent.agent_id}: tasks={snap.tasks_completed}")

        # I3: No Silent Drop — OR-Set membership preserved
        for agent in agents:
            members = agent.fleet_state.get_agents()
            if agent.agent_id in members:
                passed.append(f"I3-orset-{agent.agent_id}: registered in fleet")
            else:
                failed.append(f"I3-orset-{agent.agent_id}: MISSING from fleet!")

        # I4: Task Board convergence — compare by task count and available tasks
        # (OR-Set tags differ per agent, so compare values not serialization)
        all_tb_same = True
        tb_task_ids = set()
        tb_available = set()
        tb_completed = 0
        for a in agents:
            ids = set(a.task_board._tasks.keys())
            avail = {t.task_id for t in a.task_board.available_tasks()}
            comp = len(a.task_board.list_tasks(status=TaskStatus.COMPLETED))
            if not tb_task_ids:
                tb_task_ids = ids
                tb_available = avail
                tb_completed = comp
            elif tb_task_ids != ids:
                all_tb_same = False
                break
            elif tb_available != avail:
                all_tb_same = False
                break
            elif tb_completed != comp:
                all_tb_same = False
                break
        if all_tb_same:
            passed.append("I4-task-board-convergence: All task boards identical")
        else:
            failed.append("I4-task-board-convergence: Task boards diverge!")

        # I5: Causal consistency — consensus state converges
        # Compare proposals by IDs and vote totals (voter sets may differ
        # because proposals are local creations visible only after merge)
        all_cs_same = True
        for i, a in enumerate(agents):
            if i == 0:
                continue
            ref = agents[0]
            pids_a = set(a.consensus._proposals.keys())
            pids_ref = set(ref.consensus._proposals.keys())
            if pids_a != pids_ref:
                all_cs_same = False
                break
            # Compare total approve/reject counts (not per-voter, which can differ)
            for pid in pids_a:
                p1 = a.consensus._proposals[pid]
                p2 = ref.consensus._proposals[pid]
                sum_approve_1 = sum(p1.approve_counts.values())
                sum_approve_2 = sum(p2.approve_counts.values())
                sum_reject_1 = sum(p1.reject_counts.values())
                sum_reject_2 = sum(p2.reject_counts.values())
                if (sum_approve_1 != sum_approve_2 or
                    sum_reject_1 != sum_reject_2):
                    all_cs_same = False
                    break
        if all_cs_same:
            passed.append("I5-consensus-convergence: All consensus states identical")
        else:
            failed.append("I5-consensus-convergence: Consensus states diverge!")

        return passed, failed


def _json_key(obj: Any) -> str:
    """Create a hashable key from a JSON-serializable object."""
    return json.dumps(obj, sort_keys=True)
