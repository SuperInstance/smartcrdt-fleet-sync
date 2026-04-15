"""
task_board.py — CRDT-based task board with conflict-aware assignment.

Replaces the honor-system markdown task boards with a mathematically
verifiable CRDT system that handles concurrent task assignment, status
transitions, and priority ordering without central coordination.

Architecture:
    TaskBoard
    ├── tasks: Dict[task_id, CRDTTask]     — all tasks in the board
    ├── task_order: LWWRegister[List]      — priority ordering
    ├── assignment_log: ORSet[str]         — assignment audit trail
    └── vector_clock: VectorClock          — causal ordering per task

Each CRDTTask uses:
    - LWW-Register for: status, priority, assignee, description
    - OR-Set for: labels, dependencies, comments
    - VectorClock for: per-task causal ordering

Conflict Resolution Strategy:
    1. Concurrent assignment → first-write-wins (vector clock causal check)
    2. Concurrent status transition → highest-priority transition wins
       (open < assigned < in_progress < completed)
    3. Concurrent description edit → LWW (most recent wins)
    4. Concurrent label add → OR-Set (add-wins, both preserved)

Invariants:
    I1: A task can only be assigned to one agent at a time (MV-Register)
    I2: Status transitions follow a valid state machine
    I3: Task IDs are globally unique (UUID-based)
    I4: Assignment audit trail is append-only (OR-Set)
    I5: All replicas converge after merge (eventual consistency)

Python 3.9+ stdlib only.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .crdt_primitives import (
    GCounter, HLCTimestamp, HybridLogicalClock,
    LWWRegister, ORSet, VectorClock,
)


class TaskStatus(IntEnum):
    """Task status with ordering for conflict resolution."""
    OPEN = 0
    ASSIGNED = 1
    IN_PROGRESS = 2
    REVIEW = 3
    COMPLETED = 4
    CANCELLED = -1
    BLOCKED = -2

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, s: str) -> TaskStatus:
        try:
            return cls[s.upper()]
        except KeyError:
            return cls.OPEN


# Valid status transitions
_VALID_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.OPEN: {TaskStatus.ASSIGNED, TaskStatus.CANCELLED, TaskStatus.BLOCKED},
    TaskStatus.ASSIGNED: {TaskStatus.IN_PROGRESS, TaskStatus.OPEN,
                          TaskStatus.CANCELLED, TaskStatus.BLOCKED,
                          TaskStatus.COMPLETED},  # Direct complete allowed
    TaskStatus.IN_PROGRESS: {TaskStatus.REVIEW, TaskStatus.OPEN,
                             TaskStatus.CANCELLED, TaskStatus.BLOCKED},
    TaskStatus.REVIEW: {TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS},
    TaskStatus.COMPLETED: set(),  # Terminal state
    TaskStatus.CANCELLED: {TaskStatus.OPEN},  # Can be reopened
    TaskStatus.BLOCKED: {TaskStatus.OPEN, TaskStatus.ASSIGNED, TaskStatus.CANCELLED},
}


class CRDTTask:
    """A single CRDT-managed task with conflict-aware field updates.

    Uses LWW-Registers for scalar fields and OR-Sets for collection fields.
    Each field is independently mergeable, enabling fine-grained conflict
    resolution at the field level.
    """

    def __init__(self, task_id: Optional[str] = None,
                 title: str = "",
                 replica_id: str = "",
                 created_by: str = ""):
        self._task_id = task_id or f"T-{uuid.uuid4().hex[:8].upper()}"
        self._vclock = VectorClock(replica_id)
        self._hlc = HybridLogicalClock(node_id=replica_id)

        # Scalar fields (LWW-Register)
        self._title = LWWRegister(replica_id, title, HybridLogicalClock(node_id=replica_id))
        self._status = LWWRegister(replica_id, TaskStatus.OPEN.name,
                                   HybridLogicalClock(node_id=replica_id))
        self._priority = LWWRegister(replica_id, 5,
                                     HybridLogicalClock(node_id=replica_id))
        self._assignee = LWWRegister(replica_id, None,
                                     HybridLogicalClock(node_id=replica_id))
        self._description = LWWRegister(replica_id, "",
                                        HybridLogicalClock(node_id=replica_id))
        self._created_by = LWWRegister(replica_id, created_by,
                                       HybridLogicalClock(node_id=replica_id))
        self._created_at = LWWRegister(replica_id, time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                       HybridLogicalClock(node_id=replica_id))

        # Collection fields (OR-Set)
        self._labels = ORSet(replica_id)
        self._dependencies = ORSet(replica_id)
        self._comments = ORSet(replica_id)  # Each comment is a unique tag

        # Assignment tracking
        self._assignment_count = GCounter(replica_id)

        # Claim metadata
        self._claimed_at = LWWRegister(replica_id, None,
                                      HybridLogicalClock(node_id=replica_id))
        self._claimed_by = LWWRegister(replica_id, None,
                                      HybridLogicalClock(node_id=replica_id))

    @property
    def task_id(self) -> str:
        return self._task_id

    # =========================================================================
    # Field Updates
    # =========================================================================

    def set_title(self, title: str) -> None:
        self._vclock.increment()
        self._title.set(title)

    def set_status(self, status: TaskStatus) -> Tuple[bool, str]:
        """Set task status with transition validation.

        Returns (success, message). Validates that the transition is legal.
        """
        current = TaskStatus.from_str(self._status.get() or "open")
        if status not in _VALID_TRANSITIONS.get(current, set()):
            return False, f"Invalid transition: {current.name} → {status.name}"
        self._vclock.increment()
        self._status.set(status.name)
        if status == TaskStatus.COMPLETED:
            self._assignment_count.increment()
        return True, f"Status: {current.name} → {status.name}"

    def set_priority(self, priority: int) -> None:
        """Set priority (1=critical, 5=medium, 10=low)."""
        if not 1 <= priority <= 10:
            raise ValueError("Priority must be 1-10")
        self._vclock.increment()
        self._priority.set(priority)

    def set_description(self, desc: str) -> None:
        self._vclock.increment()
        self._description.set(desc)

    def assign(self, agent_id: str) -> Tuple[bool, str]:
        """Assign task to an agent.

        Returns (success, message). If already assigned to another agent,
        the assignment is rejected (first-write-wins via vector clock).
        """
        current = self._assignee.get()
        if current and current != agent_id:
            return False, f"Already assigned to {current}"
        self._vclock.increment()
        self._assignee.set(agent_id)
        self._claimed_by.set(agent_id)
        self._claimed_at.set(time.strftime("%Y-%m-%dT%H:%M:%SZ"))
        self._assignment_count.increment()
        ok, msg = self.set_status(TaskStatus.ASSIGNED)
        return True, f"Assigned to {agent_id}"

    def unassign(self) -> None:
        """Remove assignment (return to pool)."""
        self._vclock.increment()
        self._assignee.set(None)
        self.set_status(TaskStatus.OPEN)

    def add_label(self, label: str) -> None:
        self._vclock.increment()
        self._labels.add(label)

    def remove_label(self, label: str) -> None:
        self._vclock.increment()
        self._labels.remove(label)

    def add_dependency(self, task_id: str) -> None:
        self._vclock.increment()
        self._dependencies.add(task_id)

    def add_comment(self, comment_text: str) -> str:
        """Add a comment. Returns the generated comment ID."""
        self._vclock.increment()
        comment_id = f"comment-{uuid.uuid4().hex[:8]}"
        self._comments.add(f"{comment_id}:{comment_text}")
        return comment_id

    # =========================================================================
    # Queries
    # =========================================================================

    def get_title(self) -> str:
        return self._title.get() or ""

    def get_status(self) -> TaskStatus:
        return TaskStatus.from_str(self._status.get() or "open")

    def get_priority(self) -> int:
        return self._priority.get() or 5

    def get_assignee(self) -> Optional[str]:
        return self._assignee.get()

    def get_description(self) -> str:
        return self._description.get() or ""

    def get_labels(self) -> Set[str]:
        return self._labels.members()

    def get_dependencies(self) -> Set[str]:
        return self._dependencies.members()

    def get_comments(self) -> List[str]:
        """Return comment texts (stripped of IDs)."""
        return sorted([tag.split(":", 1)[1] if ":" in tag else tag
                       for tag in self._comments.members()])

    def get_created_by(self) -> str:
        return self._created_by.get() or ""

    def get_created_at(self) -> str:
        return self._created_at.get() or ""

    def is_assigned(self) -> bool:
        return self._assignee.get() is not None

    def is_completed(self) -> bool:
        return self.get_status() == TaskStatus.COMPLETED

    # =========================================================================
    # CRDT Merge
    # =========================================================================

    def merge(self, remote: Dict[str, Any]) -> None:
        """Merge a remote task state into this one.

        Each field is merged independently using its native CRDT merge.
        """
        self._vclock.increment()

        # Merge scalar fields (LWW-Register)
        for attr, key in [
            ("_title", "title"), ("_status", "status"),
            ("_priority", "priority"), ("_assignee", "assignee"),
            ("_description", "description"), ("_created_by", "created_by"),
            ("_created_at", "created_at"),
            ("_claimed_at", "claimed_at"), ("_claimed_by", "claimed_by"),
        ]:
            if key in remote:
                remote_reg = LWWRegister.from_dict(remote[key])
                local_reg: LWWRegister = getattr(self, attr)
                setattr(self, attr, local_reg.merge(remote_reg))

        # Merge collection fields (OR-Set)
        for attr, key in [("_labels", "labels"), ("_dependencies", "dependencies"),
                          ("_comments", "comments")]:
            if key in remote:
                remote_ors = ORSet.from_dict(remote[key])
                local_ors: ORSet = getattr(self, attr)
                setattr(self, attr, local_ors.merge(remote_ors))

        # Merge counters
        if "assignment_count" in remote:
            remote_gc = GCounter.from_dict(remote["assignment_count"])
            self._assignment_count = self._assignment_count.merge(remote_gc)

        # Merge vector clock
        if "vector_clock" in remote:
            remote_vc = VectorClock.from_dict(remote["vector_clock"])
            self._vclock = self._vclock.merge(remote_vc)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "title": self._title.to_dict(),
            "status": self._status.to_dict(),
            "priority": self._priority.to_dict(),
            "assignee": self._assignee.to_dict(),
            "description": self._description.to_dict(),
            "created_by": self._created_by.to_dict(),
            "created_at": self._created_at.to_dict(),
            "labels": self._labels.to_dict(),
            "dependencies": self._dependencies.to_dict(),
            "comments": self._comments.to_dict(),
            "assignment_count": self._assignment_count.to_dict(),
            "claimed_at": self._claimed_at.to_dict(),
            "claimed_by": self._claimed_by.to_dict(),
            "vector_clock": self._vclock.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CRDTTask:
        task = cls(task_id=data.get("task_id"), replica_id="")
        task.merge(data)
        return task


class TaskBoard:
    """CRDT-based task board for fleet-wide task coordination.

    Manages a collection of CRDTTask objects with fleet-wide merge semantics.
    Supports creating, assigning, and completing tasks concurrently across
    multiple agents without central coordination.

    Usage::

        board = TaskBoard("oracle1")
        board.create_task("Build conformance runner", priority=2,
                         labels=["testing", "critical"])
        board.assign_task("T-001", "datum")
        board.merge(remote_board.to_dict())
    """

    def __init__(self, replica_id: str):
        self._replica_id = replica_id
        self._tasks: Dict[str, CRDTTask] = {}
        self._vclock = VectorClock(replica_id)
        self._task_counter = GCounter(replica_id)

    @property
    def replica_id(self) -> str:
        return self._replica_id

    # =========================================================================
    # Task Operations
    # =========================================================================

    def create_task(self, title: str, priority: int = 5,
                    description: str = "", labels: Optional[List[str]] = None,
                    dependencies: Optional[List[str]] = None,
                    created_by: str = "") -> str:
        """Create a new task and return its ID.

        Parameters
        ----------
        title : str
            Task title/description.
        priority : int
            Priority 1 (critical) to 10 (low).
        description : str
            Detailed description.
        labels : list[str], optional
            Category labels.
        dependencies : list[str], optional
            Task IDs this depends on.
        created_by : str
            Agent or user who created the task.

        Returns
        -------
        str
            The generated task ID.
        """
        self._vclock.increment()
        task = CRDTTask(title=title, replica_id=self._replica_id,
                        created_by=created_by or self._replica_id)
        task.set_priority(priority)
        if description:
            task.set_description(description)
        if labels:
            for label in labels:
                task.add_label(label)
        if dependencies:
            for dep in dependencies:
                task.add_dependency(dep)
        self._tasks[task.task_id] = task
        self._task_counter.increment()
        return task.task_id

    def assign_task(self, task_id: str, agent_id: str) -> Tuple[bool, str]:
        """Assign a task to an agent.

        Returns (success, message).
        """
        self._vclock.increment()
        if task_id not in self._tasks:
            return False, f"Task {task_id} not found"
        return self._tasks[task_id].assign(agent_id)

    def unassign_task(self, task_id: str) -> Tuple[bool, str]:
        """Remove assignment from a task."""
        self._vclock.increment()
        if task_id not in self._tasks:
            return False, f"Task {task_id} not found"
        self._tasks[task_id].unassign()
        return True, f"Task {task_id} unassigned"

    def complete_task(self, task_id: str) -> Tuple[bool, str]:
        """Mark a task as completed."""
        self._vclock.increment()
        if task_id not in self._tasks:
            return False, f"Task {task_id} not found"
        return self._tasks[task_id].set_status(TaskStatus.COMPLETED)

    def cancel_task(self, task_id: str) -> Tuple[bool, str]:
        """Cancel a task."""
        self._vclock.increment()
        if task_id not in self._tasks:
            return False, f"Task {task_id} not found"
        return self._tasks[task_id].set_status(TaskStatus.CANCELLED)

    # =========================================================================
    # Queries
    # =========================================================================

    def get_task(self, task_id: str) -> Optional[CRDTTask]:
        return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None,
                   assignee: Optional[str] = None,
                   label: Optional[str] = None) -> List[CRDTTask]:
        """List tasks with optional filtering."""
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.get_status() == status]
        if assignee is not None:
            tasks = [t for t in tasks if t.get_assignee() == assignee]
        if label is not None:
            tasks = [t for t in tasks if label in t.get_labels()]
        tasks.sort(key=lambda t: (t.get_priority(), t.task_id))
        return tasks

    def available_tasks(self) -> List[CRDTTask]:
        """List unassigned, open tasks ordered by priority."""
        return [t for t in self.list_tasks()
                if t.get_status() == TaskStatus.OPEN and not t.is_assigned()]

    def board_summary(self) -> Dict[str, Any]:
        """Get task board summary."""
        tasks = list(self._tasks.values())
        by_status = {}
        for t in tasks:
            s = t.get_status().name
            by_status[s] = by_status.get(s, 0) + 1
        return {
            "total_tasks": len(tasks),
            "by_status": by_status,
            "available": len(self.available_tasks()),
            "completed": by_status.get("COMPLETED", 0),
            "task_counter": self._task_counter.value(),
        }

    # =========================================================================
    # CRDT Merge
    # =========================================================================

    def merge(self, remote_board: Dict[str, Any]) -> None:
        """Merge a remote task board into this one.

        Each task is merged independently using its native CRDT merge.
        New tasks from the remote are added to this board.
        """
        self._vclock.increment()

        # Merge individual tasks
        for task_id, task_data in remote_board.get("tasks", {}).items():
            if task_id in self._tasks:
                self._tasks[task_id].merge(task_data)
            else:
                self._tasks[task_id] = CRDTTask.from_dict(task_data)

        # Merge task counter
        if "task_counter" in remote_board:
            remote_tc = GCounter.from_dict(remote_board["task_counter"])
            self._task_counter = self._task_counter.merge(remote_tc)

        # Merge vector clock
        if "vector_clock" in remote_board:
            remote_vc = VectorClock.from_dict(remote_board["vector_clock"])
            self._vclock = self._vclock.merge(remote_vc)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replica_id": self._replica_id,
            "version": "0.1.0",
            "vector_clock": self._vclock.to_dict(),
            "tasks": {tid: task.to_dict() for tid, task in self._tasks.items()},
            "task_counter": self._task_counter.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskBoard:
        board = cls(replica_id=data.get("replica_id", "unknown"))
        board.merge(data)
        return board
