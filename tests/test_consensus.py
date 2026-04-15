"""
test_consensus.py — Tests for emergence-based fleet consensus engine.
"""

from smartcrdt_fleet.consensus import ConsensusEngine, Vote, ProposalStatus


class TestConsensusProposal:
    def test_create_proposal(self):
        engine = ConsensusEngine("r1")
        pid = engine.propose("Deploy v2", "Critical update")
        assert pid.startswith("PROP-")
        prop = engine.get_proposal(pid)
        assert prop is not None
        assert prop.subject == "Deploy v2"
        assert prop.status == "open"

    def test_vote(self):
        engine = ConsensusEngine("r1")
        engine.set_fleet_size(3)
        pid = engine.propose("Test proposal")
        ok, msg = engine.vote(pid, "r1", Vote.APPROVE)
        assert ok
        prop = engine.get_proposal(pid)
        assert "r1" in prop.voters

    def test_double_vote_rejected(self):
        engine = ConsensusEngine("r1")
        engine.set_fleet_size(3)
        pid = engine.propose("Test")
        engine.vote(pid, "r1", Vote.APPROVE)
        ok, _ = engine.vote(pid, "r1", Vote.REJECT)
        assert not ok

    def test_approval(self):
        engine = ConsensusEngine("r1")
        engine.set_fleet_size(3)
        pid = engine.propose("Test")
        engine.vote(pid, "r1", Vote.APPROVE)
        engine.vote(pid, "r2", Vote.APPROVE)
        status, reason = engine.resolve(pid)
        assert status == ProposalStatus.APPROVED

    def test_rejection(self):
        engine = ConsensusEngine("r1")
        engine.set_fleet_size(3)
        pid = engine.propose("Test")
        engine.vote(pid, "r1", Vote.REJECT)
        engine.vote(pid, "r2", Vote.REJECT)
        status, reason = engine.resolve(pid)
        assert status == ProposalStatus.REJECTED

    def test_list_proposals(self):
        engine = ConsensusEngine("r1")
        engine.propose("First")
        engine.propose("Second")
        engine.propose("Third")
        assert len(engine.list_proposals()) == 3

    def test_list_by_status(self):
        engine = ConsensusEngine("r1")
        engine.set_fleet_size(3)
        p1 = engine.propose("Open")
        p2 = engine.propose("To Approve")
        engine.vote(p2, "r1", Vote.APPROVE)
        engine.vote(p2, "r2", Vote.APPROVE)
        engine.resolve(p2)
        assert len(engine.list_proposals(status="open")) == 1
        assert len(engine.list_proposals(status="approved")) == 1


class TestConsensusMerge:
    def test_merge_votes(self):
        e1 = ConsensusEngine("r1")
        e1.set_fleet_size(3)
        e2 = ConsensusEngine("r2")
        pid = e1.propose("Merge test")
        e1.vote(pid, "r1", Vote.APPROVE)
        e2.merge(e1.to_dict())
        e2.vote(pid, "r2", Vote.APPROVE)
        e1.merge(e2.to_dict())
        # Both should have both votes
        prop = e1.get_proposal(pid)
        assert "r1" in prop.voters
        assert "r2" in prop.voters

    def test_bidirectional_convergence(self):
        e1 = ConsensusEngine("r1")
        e2 = ConsensusEngine("r2")
        e1.set_fleet_size(4)
        e2.set_fleet_size(4)
        p1 = e1.propose("Shared proposal")
        e1.vote(p1, "r1", Vote.APPROVE)
        e2.merge(e1.to_dict())
        e2.vote(p1, "r2", Vote.APPROVE)
        e1.merge(e2.to_dict())
        # Both should converge
        s1 = e1.get_proposal(p1).voters
        s2 = e2.get_proposal(p1).voters
        assert s1 == s2

    def test_serialization_roundtrip(self):
        engine = ConsensusEngine("r1", quorum_fraction=0.6)
        engine.set_fleet_size(5)
        pid = engine.propose("Test")
        engine.vote(pid, "r1", Vote.APPROVE)
        data = engine.to_dict()
        restored = ConsensusEngine.from_dict(data)
        prop = restored.get_proposal(pid)
        assert prop is not None
        assert "r1" in prop.voters
