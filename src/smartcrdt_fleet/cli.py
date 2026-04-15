"""
cli.py — Command-line interface for smartcrdt-fleet-sync.

Provides fleet management commands for working with CRDT state:

    python -m smartcrdt_fleet.cli init         Initialize .fleet-crdt/ in a repo
    python -m smartcrdt_fleet.cli status       Show fleet state summary
    python -m smartcrdt_fleet.cli board        Show task board summary
    python -m smartcrdt_fleet.cli propose      Create a fleet proposal
    python -m smartcrdt_fleet.cli simulate     Run fleet simulation
    python -m smartcrdt_fleet.cli test         Run property-based tests

Python 3.9+ stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from .consensus import ConsensusEngine, Vote
from .fleet_state import FleetState
from .git_state_store import GitStateStore
from .task_board import TaskBoard
from .simulation import FleetSimulator


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize .fleet-crdt/ directory in a repository."""
    repo_root = Path(args.repo_root or ".").resolve()
    store = GitStateStore(str(repo_root), args.agent_id or "unknown")
    store.ensure_crdt_dir()

    # Initialize fleet state with this agent
    fs = store.load_fleet_state()
    fs.register_agent(args.agent_id or "unknown", capabilities=args.capabilities or [])
    store.save_fleet_state(fs)

    # Initialize empty task board
    tb = store.load_task_board()
    store.save_task_board(tb)

    # Initialize empty consensus
    cs = store.load_consensus()
    store.save_consensus(cs)

    print(f"Initialized .fleet-crdt/ in {repo_root}")
    print(f"  Agent: {args.agent_id or 'unknown'}")
    print(f"  Capabilities: {args.capabilities or []}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show fleet state summary."""
    repo_root = Path(args.repo_root or ".").resolve()
    store = GitStateStore(str(repo_root), args.agent_id or "unknown")
    fs = store.load_fleet_state()
    summary = fs.fleet_summary()

    print(f"Fleet Status (agent: {store.agent_id})")
    print(f"  Total agents: {summary['total_agents']}")
    print(f"  Active agents: {summary['active_agents']}")
    print(f"  Total tasks completed: {summary['total_tasks_completed']}")
    print(f"  Vector clock: {summary['fleet_vector_clock']}")
    print()
    for aid, snap in summary.get("agents", {}).items():
        status = snap.get("status", "unknown")
        health = snap.get("health_score", 0.0)
        caps = ", ".join(snap.get("capabilities", []))
        print(f"  [{status}] {aid}: health={health:.2f} caps=[{caps}]")
    return 0


def cmd_board(args: argparse.Namespace) -> int:
    """Show task board summary."""
    repo_root = Path(args.repo_root or ".").resolve()
    store = GitStateStore(str(repo_root), args.agent_id or "unknown")
    tb = store.load_task_board()
    summary = tb.board_summary()

    print(f"Task Board (agent: {store.agent_id})")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Available: {summary['available']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  By status: {summary['by_status']}")
    print()

    if args.show_all:
        for task in tb.list_tasks():
            status = task.get_status().name
            assignee = task.get_assignee() or "unassigned"
            labels = ", ".join(task.get_labels())
            print(f"  [{status}] {task.task_id}: {task.get_title()[:60]} "
                  f"(p={task.get_priority()}, {assignee}) [{labels}]")
    else:
        for task in tb.available_tasks():
            print(f"  [OPEN] {task.task_id}: {task.get_title()[:60]} "
                  f"(p={task.get_priority()})")
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    """Create a fleet proposal."""
    repo_root = Path(args.repo_root or ".").resolve()
    store = GitStateStore(str(repo_root), args.agent_id or "unknown")
    cs = store.load_consensus()
    pid = cs.propose(args.subject, args.description or "", expires_in_hours=args.expires)
    store.save_consensus(cs)
    print(f"Created proposal {pid}: {args.subject}")
    print(f"  Expires in {args.expires}h")
    return 0


def cmd_vote(args: argparse.Namespace) -> int:
    """Vote on a proposal."""
    repo_root = Path(args.repo_root or ".").resolve()
    store = GitStateStore(str(repo_root), args.agent_id or "unknown")
    cs = store.load_consensus()
    ok, msg = cs.vote(args.proposal_id, args.agent_id or "unknown",
                     Vote(args.vote))
    store.save_consensus(cs)
    if ok:
        print(f"Voted {args.vote} on {args.proposal_id}")
    else:
        print(f"Vote failed: {msg}", file=sys.stderr)
        return 1
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run fleet simulation."""
    sim = FleetSimulator(seed=args.seed)

    if args.sim_type == "partition":
        result = sim.run_partition_simulation(
            num_agents=args.agents, num_operations=args.operations,
        )
    elif args.sim_type == "message_loss":
        result = sim.run_message_loss_simulation(
            num_agents=args.agents, num_rounds=args.operations,
            loss_probability=args.loss_prob,
        )
    elif args.sim_type == "clock_skew":
        result = sim.run_clock_skew_simulation(
            num_agents=args.agents, num_operations=args.operations,
            max_skew_ms=args.skew_ms,
        )
    elif args.sim_type == "chaos":
        result = sim.run_full_chaos_simulation(
            num_agents=args.agents, num_rounds=args.operations,
        )
    else:
        print(f"Unknown simulation type: {args.sim_type}", file=sys.stderr)
        return 1

    print(result.summary())
    print()
    for step in result.steps:
        print(f"  {step}")
    if result.invariants_failed:
        print()
        print("FAILED INVARIANTS:")
        for inv in result.invariants_failed:
            print(f"  ✗ {inv}")
    return 0 if result.convergence_achieved else 1


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge remote fleet state."""
    repo_root = Path(args.repo_root or ".").resolve()
    store = GitStateStore(str(repo_root), args.agent_id or "unknown")
    summary = store.full_sync(args.remote_dirs or [])
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="smartcrdt-fleet",
        description="CRDT-based multi-agent fleet coordination",
    )
    parser.add_argument("--agent-id", "-a", default="unknown",
                        help="Agent identity for CRDT operations")
    parser.add_argument("--repo-root", "-r", default=".",
                        help="Repository root path")
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Initialize .fleet-crdt/")
    p_init.add_argument("--capabilities", "-c", nargs="*", default=[])
    p_init.set_defaults(func=cmd_init)

    # status
    p_status = sub.add_parser("status", help="Show fleet state")
    p_status.set_defaults(func=cmd_status)

    # board
    p_board = sub.add_parser("board", help="Show task board")
    p_board.add_argument("--show-all", action="store_true", help="Show all tasks")
    p_board.set_defaults(func=cmd_board)

    # propose
    p_propose = sub.add_parser("propose", help="Create proposal")
    p_propose.add_argument("subject", help="Proposal subject")
    p_propose.add_argument("--description", "-d", default="")
    p_propose.add_argument("--expires", type=int, default=72, help="Expires in hours")
    p_propose.set_defaults(func=cmd_propose)

    # vote
    p_vote = sub.add_parser("vote", help="Vote on proposal")
    p_vote.add_argument("proposal_id", help="Proposal ID")
    p_vote.add_argument("vote", choices=["approve", "reject", "abstain"])
    p_vote.set_defaults(func=cmd_vote)

    # simulate
    p_sim = sub.add_parser("simulate", help="Run fleet simulation")
    p_sim.add_argument("sim_type",
                       choices=["partition", "message_loss", "clock_skew", "chaos"])
    p_sim.add_argument("--agents", type=int, default=5)
    p_sim.add_argument("--operations", type=int, default=10)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--loss-prob", type=float, default=0.3)
    p_sim.add_argument("--skew-ms", type=int, default=5000)
    p_sim.set_defaults(func=cmd_simulate)

    # merge
    p_merge = sub.add_parser("merge", help="Merge remote state")
    p_merge.add_argument("remote_dirs", nargs="*", help="Remote repo paths")
    p_merge.set_defaults(func=cmd_merge)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
