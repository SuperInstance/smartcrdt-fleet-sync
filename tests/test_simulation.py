"""
test_simulation.py — Fleet simulation tests under partitions, loss, skew.
"""

from smartcrdt_fleet.simulation import FleetSimulator


class TestPartitionSimulation:
    def test_basic_partition_converges(self):
        sim = FleetSimulator(seed=42)
        result = sim.run_partition_simulation(
            num_agents=5,
            num_operations=10,
            partition_groups=[[0, 1, 2], [3, 4]],
        )
        print(result.summary())
        assert result.convergence_achieved, \
            f"Partition simulation failed: {result.invariants_failed}"

    def test_single_agent_partition(self):
        sim = FleetSimulator(seed=123)
        result = sim.run_partition_simulation(
            num_agents=3,
            num_operations=5,
            partition_groups=[[0], [1, 2]],
        )
        assert result.convergence_achieved

    def test_many_agents(self):
        sim = FleetSimulator(seed=456)
        result = sim.run_partition_simulation(
            num_agents=10,
            num_operations=15,
            partition_groups=[
                list(range(0, 3)),
                list(range(3, 6)),
                list(range(6, 10)),
            ],
        )
        assert result.convergence_achieved


class TestMessageLossSimulation:
    def test_moderate_loss_converges(self):
        sim = FleetSimulator(seed=42)
        result = sim.run_message_loss_simulation(
            num_agents=5,
            num_rounds=10,
            loss_probability=0.3,
        )
        print(result.summary())
        assert result.convergence_achieved, \
            f"Message loss simulation failed: {result.invariants_failed}"

    def test_heavy_loss_converges(self):
        sim = FleetSimulator(seed=789)
        result = sim.run_message_loss_simulation(
            num_agents=4,
            num_rounds=8,
            loss_probability=0.6,
        )
        assert result.convergence_achieved

    def test_no_loss(self):
        sim = FleetSimulator(seed=101)
        result = sim.run_message_loss_simulation(
            num_agents=3,
            num_rounds=5,
            loss_probability=0.0,
        )
        assert result.convergence_achieved


class TestClockSkewSimulation:
    def test_moderate_skew(self):
        sim = FleetSimulator(seed=42)
        result = sim.run_clock_skew_simulation(
            num_agents=5,
            num_operations=15,
            max_skew_ms=1000,
        )
        print(result.summary())
        assert result.convergence_achieved, \
            f"Clock skew simulation failed: {result.invariants_failed}"

    def test_extreme_skew(self):
        sim = FleetSimulator(seed=202)
        result = sim.run_clock_skew_simulation(
            num_agents=4,
            num_operations=10,
            max_skew_ms=10000,
        )
        assert result.convergence_achieved


class TestFullChaosSimulation:
    def test_chaos_converges(self):
        sim = FleetSimulator(seed=42)
        result = sim.run_full_chaos_simulation(
            num_agents=7,
            num_rounds=10,
            loss_probability=0.2,
            partition_probability=0.3,
            max_skew_ms=3000,
        )
        print(result.summary())
        assert result.convergence_achieved, \
            f"Chaos simulation failed: {result.invariants_failed}"

    def test_heavy_chaos(self):
        sim = FleetSimulator(seed=303)
        result = sim.run_full_chaos_simulation(
            num_agents=10,
            num_rounds=15,
            loss_probability=0.4,
            partition_probability=0.5,
            max_skew_ms=5000,
        )
        assert result.convergence_achieved


class TestConvergenceAcrossSimulations:
    """Property-based tests: verify invariants hold across many random seeds."""

    def test_partition_convergence_many_seeds(self):
        for seed in range(10):
            sim = FleetSimulator(seed=seed)
            result = sim.run_partition_simulation(
                num_agents=5, num_operations=10,
            )
            assert result.convergence_achieved, \
                f"Seed {seed}: {result.invariants_failed}"

    def test_message_loss_convergence_many_seeds(self):
        for seed in range(10):
            sim = FleetSimulator(seed=seed)
            result = sim.run_message_loss_simulation(
                num_agents=4, num_rounds=8, loss_probability=0.3,
            )
            assert result.convergence_achieved, \
                f"Seed {seed}: {result.invariants_failed}"

    def test_chaos_convergence_many_seeds(self):
        for seed in range(5):
            sim = FleetSimulator(seed=seed)
            result = sim.run_full_chaos_simulation(
                num_agents=6, num_rounds=10,
            )
            assert result.convergence_achieved, \
                f"Seed {seed}: {result.invariants_failed}"
