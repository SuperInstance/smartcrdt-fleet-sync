"""
git_state_store.py — JSON-backed CRDT state persistence for git-native coordination.

Persists CRDT state (fleet state, task board, consensus) as JSON files that
are committed to git. Each agent reads peer state from committed JSON files,
merges using CRDT operations, and writes back the merged result.

File Layout (within a repo)::

    .fleet-crdt/
        fleet-state.json     — FleetState CRDT
        task-board.json      — TaskBoard CRDT
        consensus.json       — ConsensusEngine CRDT
        agent-{id}.json      — Per-agent state deltas
        merge-log.json       — Audit log of all merges

Merge Protocol:
    1. Agent reads .fleet-crdt/ from latest commit
    2. Agent applies local operations to its CRDT replicas
    3. Agent pulls latest remote changes (git pull)
    4. Agent reads remote .fleet-crdt/ files
    5. Agent merges remote state into local CRDTs (CRDT merge — always safe)
    6. Agent writes merged state back to .fleet-crdt/
    7. Agent commits and pushes

The CRDT merge guarantees convergence regardless of concurrent writes.

Python 3.9+ stdlib only.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .consensus import ConsensusEngine
from .fleet_state import FleetState
from .task_board import TaskBoard


# State file names
_FLEET_STATE_FILE = "fleet-state.json"
_TASK_BOARD_FILE = "task-board.json"
_CONSENSUS_FILE = "consensus.json"
_MERGE_LOG_FILE = "merge-log.json"
_CRDT_DIR = ".fleet-crdt"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, content: str) -> None:
    """Write content atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


class MergeLogEntry:
    """Audit log entry for a CRDT merge operation."""

    def __init__(self, agent_id: str, source: str, component: str,
                 remote_agent: str = "", details: str = ""):
        self.agent_id = agent_id
        self.source = source
        self.component = component
        self.remote_agent = remote_agent
        self.details = details
        self.timestamp = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "source": self.source,
            "component": self.component,
            "remote_agent": self.remote_agent,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class GitStateStore:
    """Git-backed state store for CRDT fleet coordination.

    Manages the lifecycle of CRDT state files: read, merge, write, audit.

    Usage::

        store = GitStateStore("/path/to/repo", "oracle1")
        fleet_state = store.load_fleet_state()
        fleet_state.register_agent("oracle1", ["coordination"])
        store.save_fleet_state(fleet_state, commit_message="Register oracle1")
    """

    def __init__(self, repo_root: str, agent_id: str):
        self._root = Path(repo_root).resolve()
        self._agent_id = agent_id
        self._crdt_dir = self._root / _CRDT_DIR

    @property
    def crdt_dir(self) -> Path:
        return self._crdt_dir

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def ensure_crdt_dir(self) -> None:
        """Create .fleet-crdt/ directory if it doesn't exist."""
        self._crdt_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Fleet State
    # =========================================================================

    def load_fleet_state(self) -> FleetState:
        """Load fleet state from JSON file, or return empty state."""
        path = self._crdt_dir / _FLEET_STATE_FILE
        if not path.is_file():
            return FleetState(self._agent_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return FleetState.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            # Corrupted file — return empty state (CRDT merge will fix it)
            return FleetState(self._agent_id)

    def save_fleet_state(self, state: FleetState) -> Path:
        """Save fleet state to JSON file."""
        self.ensure_crdt_dir()
        path = self._crdt_dir / _FLEET_STATE_FILE
        _atomic_write(path, json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
        return path

    # =========================================================================
    # Task Board
    # =========================================================================

    def load_task_board(self) -> TaskBoard:
        """Load task board from JSON file, or return empty board."""
        path = self._crdt_dir / _TASK_BOARD_FILE
        if not path.is_file():
            return TaskBoard(self._agent_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TaskBoard.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return TaskBoard(self._agent_id)

    def save_task_board(self, board: TaskBoard) -> Path:
        """Save task board to JSON file."""
        self.ensure_crdt_dir()
        path = self._crdt_dir / _TASK_BOARD_FILE
        _atomic_write(path, json.dumps(board.to_dict(), indent=2, ensure_ascii=False))
        return path

    # =========================================================================
    # Consensus
    # =========================================================================

    def load_consensus(self) -> ConsensusEngine:
        """Load consensus engine from JSON file, or return empty engine."""
        path = self._crdt_dir / _CONSENSUS_FILE
        if not path.is_file():
            return ConsensusEngine(self._agent_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ConsensusEngine.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return ConsensusEngine(self._agent_id)

    def save_consensus(self, engine: ConsensusEngine) -> Path:
        """Save consensus engine to JSON file."""
        self.ensure_crdt_dir()
        path = self._crdt_dir / _CONSENSUS_FILE
        _atomic_write(path, json.dumps(engine.to_dict(), indent=2, ensure_ascii=False))
        return path

    # =========================================================================
    # Merge Protocol
    # =========================================================================

    def merge_remote_state(self, remote_repo_root: str) -> List[MergeLogEntry]:
        """Merge remote CRDT state into local state.

        Reads remote .fleet-crdt/ files, merges into local CRDTs,
        and saves the merged result. Returns merge log entries.

        This is the core git-based coordination primitive.
        """
        remote_dir = Path(remote_repo_root).resolve() / _CRDT_DIR
        log_entries: List[MergeLogEntry] = []

        if not remote_dir.is_dir():
            return log_entries

        # Merge fleet state
        remote_fs_path = remote_dir / _FLEET_STATE_FILE
        if remote_fs_path.is_file():
            try:
                remote_data = json.loads(remote_fs_path.read_text(encoding="utf-8"))
                if remote_data.get("replica_id") != self._agent_id:
                    local_fs = self.load_fleet_state()
                    local_fs.merge(remote_data)
                    self.save_fleet_state(local_fs)
                    log_entries.append(MergeLogEntry(
                        agent_id=self._agent_id,
                        source="remote",
                        component="fleet_state",
                        remote_agent=remote_data.get("replica_id", "unknown"),
                        details=f"Merged fleet state from {remote_data.get('replica_id')}",
                    ))
            except (json.JSONDecodeError, TypeError):
                pass

        # Merge task board
        remote_tb_path = remote_dir / _TASK_BOARD_FILE
        if remote_tb_path.is_file():
            try:
                remote_data = json.loads(remote_tb_path.read_text(encoding="utf-8"))
                if remote_data.get("replica_id") != self._agent_id:
                    local_tb = self.load_task_board()
                    local_tb.merge(remote_data)
                    self.save_task_board(local_tb)
                    log_entries.append(MergeLogEntry(
                        agent_id=self._agent_id,
                        source="remote",
                        component="task_board",
                        remote_agent=remote_data.get("replica_id", "unknown"),
                        details=f"Merged task board from {remote_data.get('replica_id')}",
                    ))
            except (json.JSONDecodeError, TypeError):
                pass

        # Merge consensus
        remote_cs_path = remote_dir / _CONSENSUS_FILE
        if remote_cs_path.is_file():
            try:
                remote_data = json.loads(remote_cs_path.read_text(encoding="utf-8"))
                if remote_data.get("replica_id") != self._agent_id:
                    local_cs = self.load_consensus()
                    local_cs.merge(remote_data)
                    self.save_consensus(local_cs)
                    log_entries.append(MergeLogEntry(
                        agent_id=self._agent_id,
                        source="remote",
                        component="consensus",
                        remote_agent=remote_data.get("replica_id", "unknown"),
                        details=f"Merged consensus from {remote_data.get('replica_id')}",
                    ))
            except (json.JSONDecodeError, TypeError):
                pass

        # Write merge log
        if log_entries:
            self._append_merge_log(log_entries)

        return log_entries

    def _append_merge_log(self, entries: List[MergeLogEntry]) -> None:
        """Append merge entries to the audit log."""
        self.ensure_crdt_dir()
        log_path = self._crdt_dir / _MERGE_LOG_FILE
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            existing = []
        existing.extend(e.to_dict() for e in entries)
        # Keep last 1000 entries
        if len(existing) > 1000:
            existing = existing[-1000:]
        _atomic_write(log_path, json.dumps(existing, indent=2, ensure_ascii=False))

    def load_merge_log(self) -> List[Dict[str, Any]]:
        """Load the merge audit log."""
        log_path = self._crdt_dir / _MERGE_LOG_FILE
        if not log_path.is_file():
            return []
        try:
            return json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    # =========================================================================
    # Full Sync (Read + Merge + Save)
    # =========================================================================

    def full_sync(self, remote_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform a full sync: merge all remote states and return summary.

        Parameters
        ----------
        remote_dirs : list[str], optional
            Paths to remote repo clones. If None, uses only local state.

        Returns
        -------
        dict
            Sync summary with merge count, components merged, and current state.
        """
        total_merges = 0
        components_merged: Dict[str, int] = {}

        if remote_dirs:
            for remote_dir in remote_dirs:
                entries = self.merge_remote_state(remote_dir)
                total_merges += len(entries)
                for entry in entries:
                    components_merged[entry.component] = \
                        components_merged.get(entry.component, 0) + 1

        fleet_state = self.load_fleet_state()
        task_board = self.load_task_board()
        consensus = self.load_consensus()

        return {
            "agent_id": self._agent_id,
            "total_merges": total_merges,
            "components_merged": components_merged,
            "fleet_summary": fleet_state.fleet_summary(),
            "board_summary": task_board.board_summary(),
            "active_proposals": len(consensus.list_proposals(status="open")),
            "crdt_dir": str(self._crdt_dir),
        }

    # =========================================================================
    # Git Integration Helpers
    # =========================================================================

    def get_changed_files(self) -> List[str]:
        """List CRDT state files that have been modified."""
        changed = []
        for fname in [_FLEET_STATE_FILE, _TASK_BOARD_FILE,
                      _CONSENSUS_FILE, _MERGE_LOG_FILE]:
            if (self._crdt_dir / fname).is_file():
                changed.append(str(self._crdt_dir / fname))
        return changed

    def state_exists(self) -> bool:
        """Check if any CRDT state files exist."""
        return self._crdt_dir.is_dir() and any(
            (self._crdt_dir / f).is_file()
            for f in [_FLEET_STATE_FILE, _TASK_BOARD_FILE, _CONSENSUS_FILE]
        )
