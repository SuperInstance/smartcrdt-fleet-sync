"""
smartcrdt_fleet — CRDT-based multi-agent fleet collaboration.

Zero external dependencies. Python 3.9+ stdlib only.

Components:
    crdt_primitives   — G-Counter, PN-Counter, LWW-Register, OR-Set, Vector Clock, HLC
    fleet_state       — FleetState CRDT for agent status, health, capabilities
    task_board        — CRDT Task Board with conflict-aware assignment
    consensus         — Emergence-based fleet consensus engine
    git_state_store   — JSON-backed state persistence (git-native)
    simulation        — Fleet simulation under partitions, loss, skew
"""

__version__ = "0.1.0"
__author__ = "Super Z — Pelagic AI Fleet Agent"
