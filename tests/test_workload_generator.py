"""Tests for env/workload_generator.py."""

import pytest
import numpy as np

from env.workload_generator import WorkloadGenerator, Task

T = 8
N = 100


@pytest.fixture(scope="module")
def tasks_uniform():
    return WorkloadGenerator(seed=42).sample_tasks(N, T, pattern="uniform")


# 1. Returns exactly N tasks
def test_task_count(tasks_uniform):
    assert len(tasks_uniform) == N


# 2. All task_ids are unique
def test_task_ids_unique(tasks_uniform):
    ids = [t.task_id for t in tasks_uniform]
    assert len(set(ids)) == N


# 3. cpu_req in [0.05, 8.0]
def test_cpu_req_range(tasks_uniform):
    for t in tasks_uniform:
        assert 0.05 <= t.cpu_req <= 8.0, f"cpu_req={t.cpu_req} out of range"


# 4. mem_req in [0.5, 4.0]
def test_mem_req_range(tasks_uniform):
    for t in tasks_uniform:
        assert 0.5 <= t.mem_req <= 4.0, f"mem_req={t.mem_req} out of range"


# 5. All priorities in {2,3,4,5,6,7,8}
def test_priority_range(tasks_uniform):
    valid = {2, 3, 4, 5, 6, 7, 8}
    for t in tasks_uniform:
        assert t.priority in valid, f"priority={t.priority} not in {valid}"


# 6. All t_start in [0, T-2]
def test_t_start_range(tasks_uniform):
    for t in tasks_uniform:
        assert 0 <= t.t_start <= T - 2, f"t_start={t.t_start} out of [0, {T-2}]"


# 7. All t_deadline satisfies t_start <= t_deadline <= T-1
def test_t_deadline_range(tasks_uniform):
    for t in tasks_uniform:
        assert t.t_start <= t.t_deadline <= T - 1, (
            f"t_deadline={t.t_deadline} violates [t_start={t.t_start}, {T-1}]"
        )


# 8. Tasks sorted by t_start ascending
def test_tasks_sorted_by_t_start(tasks_uniform):
    starts = [t.t_start for t in tasks_uniform]
    assert starts == sorted(starts)


# 9. SLA values consistent with priority=8
def test_sla_from_priority_8(tasks_uniform):
    p8_tasks = [t for t in tasks_uniform if t.priority == 8]
    assert len(p8_tasks) > 0, "No priority-8 task found in 100-task sample"
    t = p8_tasks[0]
    assert t.sla_latency_ms == pytest.approx(44.0, abs=0.01)
    assert t.sla_reliability == pytest.approx(0.98, abs=1e-9)
    assert t.sla_throughput_rps == pytest.approx(130.0, abs=1e-9)


# 10. Same seed → identical tasks; different seed → different tasks
def test_seed_determinism():
    tasks_a = WorkloadGenerator(seed=42).sample_tasks(N, T)
    tasks_b = WorkloadGenerator(seed=42).sample_tasks(N, T)
    tasks_c = WorkloadGenerator(seed=99).sample_tasks(N, T)

    for a, b in zip(tasks_a, tasks_b):
        assert a.cpu_req == b.cpu_req
        assert a.t_start == b.t_start

    cpu_a = [t.cpu_req for t in tasks_a]
    cpu_c = [t.cpu_req for t in tasks_c]
    assert cpu_a != cpu_c


# 11. Bursty: at least 60% of tasks have t_start in {0,1,2}
def test_bursty_pattern_concentration():
    tasks = WorkloadGenerator(seed=42).sample_tasks(N, T, pattern="bursty")
    burst_count = sum(1 for t in tasks if t.t_start in {0, 1, 2})
    assert burst_count / N >= 0.60, f"Only {burst_count}/{N} tasks in burst window"


# 12. Heavy: mean cpu_req > uniform with same seed
def test_heavy_higher_cpu():
    gen = WorkloadGenerator(seed=42)
    uniform_tasks = gen.sample_tasks(N, T, pattern="uniform")
    heavy_tasks = WorkloadGenerator(seed=42).sample_tasks(N, T, pattern="heavy")
    mean_uniform = np.mean([t.cpu_req for t in uniform_tasks])
    mean_heavy = np.mean([t.cpu_req for t in heavy_tasks])
    assert mean_heavy > mean_uniform, (
        f"heavy mean {mean_heavy:.4f} should exceed uniform mean {mean_uniform:.4f}"
    )


# 13. Light: mean cpu_req < uniform with same seed
def test_light_lower_cpu():
    uniform_tasks = WorkloadGenerator(seed=42).sample_tasks(N, T, pattern="uniform")
    light_tasks = WorkloadGenerator(seed=42).sample_tasks(N, T, pattern="light")
    mean_uniform = np.mean([t.cpu_req for t in uniform_tasks])
    mean_light = np.mean([t.cpu_req for t in light_tasks])
    assert mean_light < mean_uniform, (
        f"light mean {mean_light:.4f} should be below uniform mean {mean_uniform:.4f}"
    )
