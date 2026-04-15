# smartcrdt-fleet-sync — CRDT-Based Multi-Agent Fleet Collaboration

> **Version 0.1.0** | **Zero Dependencies** (Python 3.9+ stdlib only)
> **108 tests, all passing** | **T-017: SmartCRDT Multi-Agent Collaboration**

## What It Does

Implements CRDT (Conflict-free Replicated Data Type) primitives for coordinating autonomous agent fleets that communicate through git repos. No central server, no database, no network services — just JSON files committed to git that merge mathematically.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     smartcrdt-fleet-sync                          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ FleetState   │  │  TaskBoard   │  │ ConsensusEngine    │     │
│  │              │  │              │  │                    │     │
│  │ OR-Set       │  │ LWW-Register │  │ PN-Counter        │     │
│  │ LWW-Register │  │ OR-Set       │  │ LWW-Register       │     │
│  │ G-Counter    │  │ G-Counter    │  │ VectorClock        │     │
│  │ VectorClock  │  │ VectorClock  │  │                    │     │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘     │
│         │                │                   │                   │
│  ┌──────┴──────────────────────────────────────┴───────────┐    │
│  │              Git State Store (.fleet-crdt/*.json)        │    │
│  │  merge protocol: pull → merge CRDTs → commit → push    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Simulation Framework                          │    │
│  │  Partition | Message Loss | Clock Skew | Full Chaos     │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## CRDT Types (6 Primitives)

| Type | Merge | Guarantee | Use Case |
|------|-------|-----------|----------|
| **G-Counter** | element-wise max | Strong | Task counts, metrics |
| **PN-Counter** | two G-Counters | Strong | Balance, inventory |
| **LWW-Register** | latest timestamp wins | Strong | Status, health, heartbeat |
| **OR-Set** | tag union | Strong | Membership, capabilities |
| **VectorClock** | element-wise max | Strong | Causal ordering |
| **HLC** | physical+logical | Strong | Distributed timestamps |

## Composite Types (3 Fleet Objects)

| Component | Description | CRDT Types Used |
|-----------|-------------|-----------------|
| **FleetState** | Agent registry, status, health, capabilities | OR-Set, LWW-Register, G-Counter, VC |
| **TaskBoard** | Concurrent task assignment with conflict resolution | LWW-Register, OR-Set, G-Counter, VC |
| **ConsensusEngine** | Proposal/voting with automatic resolution | PN-Counter, LWW-Register, VC |

## Simulation Framework

Tests fleet behavior under failure conditions:

- **Partition**: Agents split into groups, operate independently, merge after healing
- **Message Loss**: Random messages dropped; verifies eventual convergence
- **Clock Skew**: HLC clocks skewed by configurable drift; verifies ordering
- **Full Chaos**: All failure modes combined; stress test

## Merge Protocol (Git-Native)

```python
from smartcrdt_fleet.git_state_store import GitStateStore

store = GitStateStore("/path/to/repo", "oracle1")

# Pull remote state and merge
store.merge_remote_state("/path/to/remote/clone")

# All CRDT merges are safe — commutative, associative, idempotent
# Even if concurrent writes happened, the result is deterministic
```

## Invariants Verified (108 Tests)

- **I1 Convergence**: All replicas reach identical state after merge
- **I2 Monotonicity**: Counters never decrease across merges
- **I3 No Silent Drop**: OR-Set preserves adds during concurrent removes
- **I4 Deterministic Resolution**: LWW always picks the same winner
- **I5 Causal Consistency**: Vector clock ordering preserved

## Quick Start

```bash
# Install (no dependencies — just Python 3.9+)
pip install -e .

# Run the full test suite
python -m pytest tests/ -v

# Run a fleet simulation
python -m smartcrdt_fleet.cli simulate partition --agents 5 --operations 20

# Initialize in a repo
python -m smartcrdt_fleet.cli init --agent-id oracle1
```

## Task Board (T-017)

Claimed from Oracle1's task board. Implements CRDT-based task coordination
replacing honor-system markdown boards with mathematically verified conflict
resolution.

- **Concurrent assignment**: First-write-wins via vector clock causal check
- **Status transitions**: Validated state machine (OPEN→ASSIGNED→IN_PROGRESS→REVIEW→COMPLETED)
- **Labels**: OR-Set add-wins semantics
- **Audit trail**: G-Counter for per-agent assignment counts

## Fleet Context

Built for the Pelagic AI fleet (SuperInstance GitHub org). Connects to the
existing message-in-a-bottle protocol and git-agent standard.

---

*Part of the SmartCRDT ecosystem. Zero external dependencies.*
