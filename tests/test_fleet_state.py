"""
test_fleet_state.py — Tests for CRDT-based fleet state synchronization.
"""

from smartcrdt_fleet.fleet_state import FleetState, AgentSnapshot
from smartcrdt_fleet.crdt_primitives import VectorClock


class TestFleetStateRegistration:
    def test_register_single_agent(self):
        fs = FleetState("r1")
        fs.register_agent("oracle1", ["coordination"])
        assert "oracle1" in fs.get_agents()

    def test_register_multiple_agents(self):
        fs = FleetState("r1")
        fs.register_agent("oracle1", ["coordination"])
        fs.register_agent("datum", ["auditing", "conformance"])
        assert len(fs.get_agents()) == 2

    def test_deregister(self):
        fs = FleetState("r1")
        fs.register_agent("temp-agent", ["test"])
        fs.deregister_agent("temp-agent")
        # Deregistered agents should still be in the registry (OR-Set)
        # with status set to "retired"
        assert "temp-agent" in fs.get_agents(), "OR-Set should keep deregistered agents"
        snap = fs.get_snapshot("temp-agent")
        assert snap is not None
        assert snap.status == "retired"

    def test_register_preserves_capabilities(self):
        fs = FleetState("r1")
        fs.register_agent("agent1", ["cap1", "cap2", "cap3"])
        snap = fs.get_snapshot("agent1")
        assert snap.capabilities == {"cap1", "cap2", "cap3"}


class TestFleetStateUpdates:
    def test_update_status(self):
        fs = FleetState("r1")
        fs.register_agent("agent1")
        fs.update_status("agent1", "active")
        assert fs.get_snapshot("agent1").status == "active"

    def test_invalid_status_raises(self):
        fs = FleetState("r1")
        fs.register_agent("agent1")
        try:
            fs.update_status("agent1", "invalid_status")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_update_health(self):
        fs = FleetState("r1")
        fs.register_agent("agent1")
        fs.update_health("agent1", 0.85)
        assert fs.get_snapshot("agent1").health_score == 0.85

    def test_health_clamp(self):
        fs = FleetState("r1")
        fs.register_agent("agent1")
        try:
            fs.update_health("agent1", 1.5)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_heartbeat(self):
        fs = FleetState("r1")
        fs.register_agent("agent1")
        fs.heartbeat("agent1")
        snap = fs.get_snapshot("agent1")
        assert snap.last_heartbeat is not None

    def test_task_count_increment(self):
        fs = FleetState("r1")
        fs.register_agent("agent1")
        fs.increment_task_count("agent1", 5)
        assert fs.get_snapshot("agent1").tasks_completed == 5
        fs.increment_task_count("agent1", 3)
        assert fs.get_snapshot("agent1").tasks_completed == 8

    def test_capability_management(self):
        fs = FleetState("r1")
        fs.register_agent("agent1", ["cap1"])
        fs.add_capability("agent1", "cap2")
        fs.remove_capability("agent1", "cap1")
        snap = fs.get_snapshot("agent1")
        assert "cap2" in snap.capabilities
        assert "cap1" not in snap.capabilities


class TestFleetStateQueries:
    def test_find_by_capability(self):
        fs = FleetState("r1")
        fs.register_agent("a1", ["rust", "gpu"])
        fs.register_agent("a2", ["python", "testing"])
        fs.register_agent("a3", ["rust", "testing"])
        rust_agents = fs.find_by_capability("rust")
        assert "a1" in rust_agents
        assert "a3" in rust_agents
        assert "a2" not in rust_agents

    def test_find_by_status(self):
        fs = FleetState("r1")
        fs.register_agent("a1")
        fs.register_agent("a2")
        fs.register_agent("a3")
        fs.update_status("a1", "active")
        fs.update_status("a2", "active")
        fs.update_status("a3", "idle")
        active = fs.find_by_status("active")
        assert len(active) == 2

    def test_fleet_summary(self):
        fs = FleetState("r1")
        fs.register_agent("a1", ["test"])
        fs.register_agent("a2", ["test"])
        fs.update_status("a1", "active")
        fs.update_status("a2", "idle")
        summary = fs.fleet_summary()
        assert summary["total_agents"] == 2
        assert summary["active_agents"] == 1

    def test_get_all_snapshots(self):
        fs = FleetState("r1")
        fs.register_agent("a1")
        fs.register_agent("a2")
        snapshots = fs.get_all_snapshots()
        assert len(snapshots) == 2
        assert "a1" in snapshots
        assert "a2" in snapshots


class TestFleetStateMerge:
    def test_convergence_after_merge(self):
        fs1 = FleetState("r1")
        fs2 = FleetState("r2")
        fs1.register_agent("a1", ["coordination"])
        fs2.register_agent("a2", ["auditing"])
        fs1.merge(fs2.to_dict())
        assert "a1" in fs1.get_agents()
        assert "a2" in fs1.get_agents()

    def test_bidirectional_convergence(self):
        fs1 = FleetState("r1")
        fs2 = FleetState("r2")
        fs1.register_agent("a1", ["cap1"])
        fs2.register_agent("a2", ["cap2"])
        fs1.update_status("a1", "active")
        fs2.update_status("a2", "idle")

        fs1.merge(fs2.to_dict())
        fs2.merge(fs1.to_dict())

        # Both should have same agents and same values (compare meaningfully)
        assert fs1.get_agents() == fs2.get_agents(), "Agent sets must converge"
        for aid in fs1.get_agents():
            s1 = fs1.get_snapshot(aid)
            s2 = fs2.get_snapshot(aid)
            assert s1 is not None and s2 is not None
            assert s1.status == s2.status, f"Agent {aid} status: {s1.status} vs {s2.status}"
            assert s1.health_score == s2.health_score, \
                f"Agent {aid} health: {s1.health_score} vs {s2.health_score}"
            assert s1.capabilities == s2.capabilities, \
                f"Agent {aid} caps: {s1.capabilities} vs {s2.capabilities}"

    def test_concurrent_capability_add_wins(self):
        fs1 = FleetState("r1")
        fs2 = FleetState("r2")
        fs1.register_agent("shared", ["cap_from_r1"])
        fs2.register_agent("shared", ["cap_from_r2"])
        fs1.merge(fs2.to_dict())
        snap = fs1.get_snapshot("shared")
        assert "cap_from_r1" in snap.capabilities
        assert "cap_from_r2" in snap.capabilities

    def test_lww_status_merge(self):
        fs1 = FleetState("r1")
        fs2 = FleetState("r2")
        fs1.register_agent("shared")
        fs2.register_agent("shared")
        fs1.update_status("shared", "active")
        import time; time.sleep(0.002)
        fs2.update_status("shared", "idle")
        fs1.merge(fs2.to_dict())
        # fs2's update is later → should win
        assert fs1.get_snapshot("shared").status == "idle"

    def test_monotonic_task_counts(self):
        fs1 = FleetState("r1")
        fs2 = FleetState("r2")
        fs1.register_agent("a1")
        fs2.register_agent("a1")
        fs1.increment_task_count("a1", 3)
        fs2.increment_task_count("a1", 5)
        fs1.merge(fs2.to_dict())
        # G-Counter: max per replica, then sum = max(3, 0) + max(0, 5) = 8
        assert fs1.get_snapshot("a1").tasks_completed >= 3
        assert fs1.get_snapshot("a1").tasks_completed >= 5

    def test_serialization_roundtrip(self):
        fs = FleetState("r1")
        fs.register_agent("a1", ["cap1", "cap2"])
        fs.update_status("a1", "active")
        fs.update_health("a1", 0.9)
        data = fs.to_dict()
        restored = FleetState.from_dict(data)
        assert "a1" in restored.get_agents()
        snap = restored.get_snapshot("a1")
        assert snap is not None
        assert snap.status == "active"

    def test_three_way_convergence(self):
        fs1 = FleetState("r1")
        fs2 = FleetState("r2")
        fs3 = FleetState("r3")
        fs1.register_agent("r1", ["coordination"])
        fs2.register_agent("r2", ["auditing"])
        fs3.register_agent("r3", ["gpu"])
        fs1.merge(fs2.to_dict())
        fs2.merge(fs3.to_dict())
        fs3.merge(fs1.to_dict())
        fs1.merge(fs3.to_dict())
        fs2.merge(fs1.to_dict())
        fs3.merge(fs2.to_dict())
        # All should converge to same agent set
        agents = {frozenset(fs.get_agents()) for fs in [fs1, fs2, fs3]}
        assert len(agents) == 1, "All three must converge to same agent set"
