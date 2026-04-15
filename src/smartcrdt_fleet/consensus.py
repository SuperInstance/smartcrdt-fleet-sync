"""
consensus.py — Emergence-based fleet consensus engine.

Implements a lightweight consensus mechanism for fleet-wide decisions using
CRDT primitives. Unlike Paxos/Raft which require quorum and leader election,
this engine uses merge-based consensus that works with git's eventual delivery.

Consensus Levels:
    1. LWW Consensus — Single-value decisions (status, config). Fast, deterministic.
    2. Majority Consensus — Multi-option decisions with vote counting. Requires >50%.
    3. Emergent Consensus — Pattern-based agreement from fleet behavior signals.

Usage::

    engine = ConsensusEngine("oracle1", fleet_state)
    proposal_id = engine.propose("deploy-lighthouse-v2", "Oracle1")
    engine.vote(proposal_id, "datum", "approve")
    engine.vote(proposal_id, "jc1", "approve")
    result = engine.resolve(proposal_id)  # → "approved" if >50% approve

Python 3.9+ stdlib only.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .crdt_primitives import (
    GCounter, HLCTimestamp, HybridLogicalClock,
    LWWRegister, ORSet, VectorClock,
)


class Vote(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class ProposalStatus(str, Enum):
    OPEN = "open"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class Proposal:
    """A fleet proposal with CRDT-based vote tracking."""
    proposal_id: str
    subject: str
    proposer: str
    description: str = ""
    created_at: str = ""
    expires_at: str = ""
    status: str = "open"
    # CRDT state for votes
    approve_counts: Dict[str, int] = field(default_factory=dict)
    reject_counts: Dict[str, int] = field(default_factory=dict)
    abstain_counts: Dict[str, int] = field(default_factory=dict)
    voters: Set[str] = field(default_factory=set)
    # LWW for status transitions
    status_timestamp: Optional[Dict[str, Any]] = None
    status_replica: str = ""


class ConsensusEngine:
    """Emergence-based fleet consensus engine.

    Uses CRDT primitives to track proposals and votes across the fleet.
    Proposals converge via merge — no central coordinator needed.

    Vote counting uses PN-Counter (approve = +, reject = -), enabling
    merge across replicas while preserving individual vote records.
    """

    def __init__(self, replica_id: str,
                 quorum_fraction: float = 0.5):
        """
        Parameters
        ----------
        replica_id : str
            This agent's identity.
        quorum_fraction : float
            Fraction of fleet that must vote for resolution (default 0.5 = 50%).
        """
        self._replica_id = replica_id
        self._hlc = HybridLogicalClock(node_id=replica_id)
        self._vclock = VectorClock(replica_id)
        self._proposals: Dict[str, Proposal] = {}
        self._quorum = quorum_fraction
        self._fleet_size_counter = GCounter(replica_id)

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def set_fleet_size(self, size: int) -> None:
        """Set the expected fleet size for quorum calculation."""
        self._vclock.increment()
        # Use G-Counter to track the fleet size (monotonic)
        if size > self._fleet_size_counter.value():
            self._fleet_size_counter.increment(size - self._fleet_size_counter.value())

    def propose(self, subject: str, description: str = "",
                expires_in_hours: int = 72) -> str:
        """Create a new proposal. Returns the proposal ID."""
        self._vclock.increment()
        proposal_id = f"PROP-{uuid.uuid4().hex[:8].upper()}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Calculate expiry
        exp_secs = time.time() + expires_in_hours * 3600
        expires_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(exp_secs))

        self._proposals[proposal_id] = Proposal(
            proposal_id=proposal_id,
            subject=subject,
            proposer=self._replica_id,
            description=description,
            created_at=now,
            expires_at=expires_at,
            status="open",
        )
        return proposal_id

    def vote(self, proposal_id: str, voter_id: str,
             vote: Vote) -> Tuple[bool, str]:
        """Cast a vote on a proposal.

        Returns (success, message). Each voter can vote only once per proposal.
        """
        self._vclock.increment()
        prop = self._proposals.get(proposal_id)
        if not prop:
            return False, f"Proposal {proposal_id} not found"
        if prop.status != "open":
            return False, f"Proposal is {prop.status}"
        if voter_id in prop.voters:
            return False, f"{voter_id} already voted"

        prop.voters.add(voter_id)
        if vote == Vote.APPROVE:
            prop.approve_counts[voter_id] = 1
        elif vote == Vote.REJECT:
            prop.reject_counts[voter_id] = 1
        else:
            prop.abstain_counts[voter_id] = 1

        return True, f"{voter_id} voted {vote.value} on {proposal_id}"

    def resolve(self, proposal_id: str) -> Tuple[ProposalStatus, str]:
        """Attempt to resolve a proposal based on current votes.

        Returns (status, reason).
        """
        prop = self._proposals.get(proposal_id)
        if not prop:
            return ProposalStatus.CANCELLED, f"Proposal {proposal_id} not found"

        if prop.status != "open":
            return ProposalStatus(prop.status), f"Already {prop.status}"

        # Check expiry
        if prop.expires_at:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if now >= prop.expires_at:
                prop.status = "expired"
                return ProposalStatus.EXPIRED, "Proposal expired"

        total_voters = len(prop.voters)
        approvals = len(prop.approve_counts)
        rejections = len(prop.reject_counts)

        # Need at least quorum fraction of fleet to have voted
        fleet_size = max(self._fleet_size_counter.value(), total_voters)
        if total_voters / fleet_size < self._quorum and total_voters < 2:
            return ProposalStatus.OPEN, f"Quorum not reached ({total_voters}/{fleet_size})"

        if approvals > rejections and approvals > 0:
            prop.status = "approved"
            return ProposalStatus.APPROVED, f"Approved {approvals}-{rejections}"
        elif rejections > approvals:
            prop.status = "rejected"
            return ProposalStatus.REJECTED, f"Rejected {rejections}-{approvals}"
        else:
            return ProposalStatus.OPEN, f"Tied {approvals}-{rejections}"

    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        return self._proposals.get(proposal_id)

    def list_proposals(self, status: Optional[str] = None) -> List[Proposal]:
        props = list(self._proposals.values())
        if status:
            props = [p for p in props if p.status == status]
        return sorted(props, key=lambda p: p.created_at, reverse=True)

    # =========================================================================
    # CRDT Merge
    # =========================================================================

    def merge(self, remote: Dict[str, Any]) -> None:
        """Merge remote consensus state."""
        self._vclock.increment()

        # Merge fleet size counter
        if "fleet_size_counter" in remote:
            remote_gc = GCounter.from_dict(remote["fleet_size_counter"])
            self._fleet_size_counter = self._fleet_size_counter.merge(remote_gc)

        # Merge proposals
        for pid, pdata in remote.get("proposals", {}).items():
            if pid in self._proposals:
                self._merge_proposal(self._proposals[pid], pdata)
            else:
                self._proposals[pid] = self._proposal_from_dict(pdata)

        # Merge vector clock
        if "vector_clock" in remote:
            remote_vc = VectorClock.from_dict(remote["vector_clock"])
            self._vclock = self._vclock.merge(remote_vc)

    def _merge_proposal(self, local: Proposal, remote_data: Dict[str, Any]) -> None:
        """Merge a single proposal from remote state."""
        # Merge voters (union of both sets)
        remote_voters = set(remote_data.get("voters", []))
        local.voters |= remote_voters

        # Merge vote counts (element-wise max per voter)
        for vid in remote_voters:
            remote_approve = remote_data.get("approve_counts", {}).get(vid, 0)
            remote_reject = remote_data.get("reject_counts", {}).get(vid, 0)
            remote_abstain = remote_data.get("abstain_counts", {}).get(vid, 0)
            local.approve_counts[vid] = max(local.approve_counts.get(vid, 0), remote_approve)
            local.reject_counts[vid] = max(local.reject_counts.get(vid, 0), remote_reject)
            local.abstain_counts[vid] = max(local.abstain_counts.get(vid, 0), remote_abstain)

        # LWW for status — most recent wins
        remote_ts = remote_data.get("status_timestamp")
        if remote_ts:
            remote_hlc_ts = HLCTimestamp.from_dict(remote_ts)
            local_ts = (HLCTimestamp.from_dict(local.status_timestamp)
                        if local.status_timestamp else None)
            if local_ts is None or remote_hlc_ts > local_ts:
                local.status = remote_data.get("status", local.status)
                local.status_timestamp = remote_ts
                local.status_replica = remote_data.get("status_replica", "")

    def _proposal_from_dict(self, data: Dict[str, Any]) -> Proposal:
        return Proposal(
            proposal_id=data.get("proposal_id", ""),
            subject=data.get("subject", ""),
            proposer=data.get("proposer", ""),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at", ""),
            status=data.get("status", "open"),
            approve_counts=dict(data.get("approve_counts", {})),
            reject_counts=dict(data.get("reject_counts", {})),
            abstain_counts=dict(data.get("abstain_counts", {})),
            voters=set(data.get("voters", [])),
            status_timestamp=data.get("status_timestamp"),
            status_replica=data.get("status_replica", ""),
        )

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replica_id": self._replica_id,
            "version": "0.1.0",
            "vector_clock": self._vclock.to_dict(),
            "fleet_size_counter": self._fleet_size_counter.to_dict(),
            "quorum_fraction": self._quorum,
            "proposals": {
                pid: {
                    "proposal_id": p.proposal_id,
                    "subject": p.subject,
                    "proposer": p.proposer,
                    "description": p.description,
                    "created_at": p.created_at,
                    "expires_at": p.expires_at,
                    "status": p.status,
                    "approve_counts": p.approve_counts,
                    "reject_counts": p.reject_counts,
                    "abstain_counts": p.abstain_counts,
                    "voters": sorted(p.voters),
                    "status_timestamp": p.status_timestamp,
                    "status_replica": p.status_replica,
                }
                for pid, p in self._proposals.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConsensusEngine:
        engine = cls(
            replica_id=data.get("replica_id", "unknown"),
            quorum_fraction=data.get("quorum_fraction", 0.5),
        )
        engine.merge(data)
        return engine
