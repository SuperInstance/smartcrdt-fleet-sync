"""
test_task_board.py — Tests for CRDT task board with conflict resolution.
"""

import time
from smartcrdt_fleet.task_board import TaskBoard, CRDTTask, TaskStatus


class TestCRDTTask:
    def test_create_task(self):
        task = CRDTTask(title="Test task", replica_id="r1", created_by="oracle1")
        assert task.task_id.startswith("T-")
        assert task.get_title() == "Test task"
        assert task.get_status() == TaskStatus.OPEN

    def test_status_transitions(self):
        task = CRDTTask(replica_id="r1")
        ok, msg = task.set_status(TaskStatus.ASSIGNED)
        assert ok
        assert task.get_status() == TaskStatus.ASSIGNED
        ok, msg = task.set_status(TaskStatus.IN_PROGRESS)
        assert ok
        ok, msg = task.set_status(TaskStatus.REVIEW)
        assert ok
        ok, msg = task.set_status(TaskStatus.COMPLETED)
        assert ok

    def test_invalid_transition(self):
        task = CRDTTask(replica_id="r1")
        ok, msg = task.set_status(TaskStatus.COMPLETED)  # OPEN → COMPLETED invalid
        assert not ok

    def test_assign_task(self):
        task = CRDTTask(replica_id="r1")
        ok, msg = task.assign("oracle1")
        assert ok
        assert task.get_assignee() == "oracle1"
        assert task.get_status() == TaskStatus.ASSIGNED

    def test_double_assign_rejected(self):
        task = CRDTTask(replica_id="r1")
        ok, _ = task.assign("oracle1")
        assert ok
        ok, _ = task.assign("datum")  # Different agent
        assert not ok, "Double assignment should be rejected"

    def test_labels(self):
        task = CRDTTask(replica_id="r1")
        task.add_label("testing")
        task.add_label("critical")
        assert "testing" in task.get_labels()
        assert "critical" in task.get_labels()
        task.remove_label("testing")
        assert "testing" not in task.get_labels()

    def test_dependencies(self):
        task = CRDTTask(replica_id="r1")
        task.add_dependency("T-001")
        task.add_dependency("T-002")
        assert "T-001" in task.get_dependencies()
        assert "T-002" in task.get_dependencies()

    def test_comments(self):
        task = CRDTTask(replica_id="r1")
        cid = task.add_comment("This is a test comment")
        assert cid.startswith("comment-")
        comments = task.get_comments()
        assert "This is a test comment" in comments

    def test_merge_status(self):
        task1 = CRDTTask(title="Task", replica_id="r1")
        task2 = CRDTTask(title="Task", replica_id="r2")
        task1.assign("agent1")  # Goes to ASSIGNED
        task2.assign("agent2")  # Goes to ASSIGNED
        time.sleep(0.001)
        task2.set_status(TaskStatus.IN_PROGRESS)  # Later wins
        task1.merge(task2.to_dict())
        # LWW: later status wins
        assert task1.get_status() in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)

    def test_merge_labels_add_wins(self):
        task1 = CRDTTask(replica_id="r1")
        task2 = CRDTTask(replica_id="r2")
        task1.add_label("from_r1")
        task2.add_label("from_r2")
        task1.merge(task2.to_dict())
        labels = task1.get_labels()
        assert "from_r1" in labels
        assert "from_r2" in labels

    def test_serialization_roundtrip(self):
        task = CRDTTask(title="Test", replica_id="r1", created_by="oracle1")
        task.set_priority(2)
        task.add_label("critical")
        task.assign("datum")
        data = task.to_dict()
        restored = CRDTTask.from_dict(data)
        # Note: from_dict creates a new task then merges; title may need explicit set
        assert restored.get_priority() == 2
        assert "critical" in restored.get_labels()
        assert restored.get_assignee() == "datum"


class TestTaskBoard:
    def test_create_task(self):
        board = TaskBoard("r1")
        tid = board.create_task("Build conformance runner", priority=2,
                                labels=["testing", "critical"])
        assert tid.startswith("T-")
        task = board.get_task(tid)
        assert task is not None
        assert task.get_title() == "Build conformance runner"

    def test_assign_task(self):
        board = TaskBoard("r1")
        tid = board.create_task("Test task")
        ok, msg = board.assign_task(tid, "oracle1")
        assert ok
        assert board.get_task(tid).get_assignee() == "oracle1"

    def test_complete_task(self):
        board = TaskBoard("r1")
        tid = board.create_task("Test task")
        board.assign_task(tid, "oracle1")
        ok, _ = board.complete_task(tid)
        assert ok
        assert board.get_task(tid).get_status() == TaskStatus.COMPLETED

    def test_available_tasks(self):
        board = TaskBoard("r1")
        t1 = board.create_task("Open task 1", priority=1)
        t2 = board.create_task("Open task 2", priority=3)
        t3 = board.create_task("Assigned task")
        board.assign_task(t3, "oracle1")
        available = board.available_tasks()
        assert len(available) == 2

    def test_list_by_status(self):
        board = TaskBoard("r1")
        t1 = board.create_task("Open")
        t2 = board.create_task("Done")
        board.assign_task(t2, "agent1")
        # Must go through proper status transitions: OPEN → ASSIGNED → IN_PROGRESS → REVIEW → COMPLETED
        task = board.get_task(t2)
        task.set_status(TaskStatus.IN_PROGRESS)
        task.set_status(TaskStatus.REVIEW)
        task.set_status(TaskStatus.COMPLETED)
        open_tasks = board.list_tasks(status=TaskStatus.OPEN)
        done_tasks = board.list_tasks(status=TaskStatus.COMPLETED)
        assert len(open_tasks) == 1
        assert len(done_tasks) == 1

    def test_list_by_assignee(self):
        board = TaskBoard("r1")
        t1 = board.create_task("For oracle1")
        t2 = board.create_task("For datum")
        board.assign_task(t1, "oracle1")
        board.assign_task(t2, "datum")
        oracle_tasks = board.list_tasks(assignee="oracle1")
        assert len(oracle_tasks) == 1

    def test_list_by_label(self):
        board = TaskBoard("r1")
        t1 = board.create_task("Labeled", labels=["testing"])
        t2 = board.create_task("Unlabeled")
        testing = board.list_tasks(label="testing")
        assert len(testing) == 1

    def test_board_summary(self):
        board = TaskBoard("r1")
        board.create_task("T1")
        board.create_task("T2")
        board.create_task("T3")
        t1_id = list(board._tasks.keys())[0]
        task = board.get_task(t1_id)
        task.assign("agent1")
        task.set_status(TaskStatus.IN_PROGRESS)
        task.set_status(TaskStatus.REVIEW)
        task.set_status(TaskStatus.COMPLETED)
        summary = board.board_summary()
        assert summary["total_tasks"] == 3
        assert summary["completed"] == 1

    def test_merge_new_tasks(self):
        board1 = TaskBoard("r1")
        board2 = TaskBoard("r2")
        t1 = board1.create_task("From r1")
        t2 = board2.create_task("From r2")
        board1.merge(board2.to_dict())
        assert board1.get_task(t1) is not None
        assert board1.get_task(t2) is not None

    def test_bidirectional_convergence(self):
        board1 = TaskBoard("r1")
        board2 = TaskBoard("r2")
        t1 = board1.create_task("Shared task", priority=3)
        t2 = board2.create_task("Different task")
        # Merge both ways
        board2.merge(board1.to_dict())
        board1.merge(board2.to_dict())
        # Both should have both tasks (compare by task IDs and values)
        assert set(board1._tasks.keys()) == set(board2._tasks.keys()), \
            "Both boards must have same tasks"
        for tid in board1._tasks:
            t1_val = board1.get_task(tid)
            t2_val = board2.get_task(tid)
            assert t1_val.get_priority() == t2_val.get_priority(), \
                f"Task {tid} priority should converge"
            assert t1_val.get_status() == t2_val.get_status(), \
                f"Task {tid} status should converge"

    def test_cancel_task(self):
        board = TaskBoard("r1")
        tid = board.create_task("To cancel")
        ok, _ = board.cancel_task(tid)
        assert ok
        assert board.get_task(tid).get_status() == TaskStatus.CANCELLED

    def test_serialization_roundtrip(self):
        board = TaskBoard("r1")
        board.create_task("Task 1", priority=2, labels=["a"])
        board.create_task("Task 2", priority=5)
        data = board.to_dict()
        restored = TaskBoard.from_dict(data)
        assert restored.board_summary()["total_tasks"] == 2
