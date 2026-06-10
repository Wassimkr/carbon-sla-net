"""Tests for env/battery_model.py."""

import numpy as np
import pytest

from env.battery_model import BatteryModel

CAPACITY = 500.0
DELTA = 0.25
GAMMA = 0.80


@pytest.fixture
def battery():
    """Fresh 500 Wh battery at 60% SoC before each test."""
    return BatteryModel(capacity_wh=CAPACITY, initial_soc_fraction=0.6,
                        delta=DELTA, gamma=GAMMA)


# 1. After reset(), soc_fraction equals initial_soc_fraction
def test_reset_soc_fraction(battery):
    battery.step(200.0, 0.0)  # disturb state
    battery.reset(0.6)
    assert battery.soc_fraction == pytest.approx(0.6)


# 2. Zero RE and zero power draw → no change
def test_zero_draw_zero_re(battery):
    initial_soc = battery.soc_wh
    result = battery.step(0.0, 0.0)
    assert result["b_charge_wh"] == pytest.approx(0.0)
    assert result["b_discharge_wh"] == pytest.approx(0.0)
    assert result["soc_wh"] == pytest.approx(initial_soc)


# 3. Large RE surplus → battery charges, does not exceed capacity
def test_charging_does_not_exceed_capacity(battery):
    # Reset to near full to make cap constraint visible
    battery.reset(0.9)
    result = battery.step(power_draw_w=0.0, re_available_wh=10000.0)
    assert result["b_charge_wh"] >= 0.0
    assert result["soc_wh"] <= CAPACITY + 1e-9


# 4. Zero RE + high power draw → discharges, bounded by delta * soc
def test_discharge_bounded_by_delta(battery):
    soc_before = battery.soc_wh
    max_discharge = DELTA * soc_before
    result = battery.step(power_draw_w=10000.0, re_available_wh=0.0)
    assert result["b_discharge_wh"] <= max_discharge + 1e-9
    assert result["b_charge_wh"] == pytest.approx(0.0)


# 5. SoC never below 0 or above capacity across 50 random steps
def test_soc_bounds(battery):
    rng = np.random.default_rng(42)
    for _ in range(50):
        power = rng.uniform(0, 300)
        re = rng.uniform(0, 400)
        result = battery.step(power, re)
        assert 0.0 <= result["soc_wh"] <= CAPACITY + 1e-9
        assert 0.0 <= result["soc_fraction"] <= 1.0 + 1e-9


# 6. Zero-capacity battery: all zeros, no exception
def test_zero_capacity_battery():
    batt = BatteryModel(capacity_wh=0.0)
    result = batt.step(100.0, 200.0)
    assert result["b_charge_wh"] == pytest.approx(0.0)
    assert result["b_discharge_wh"] == pytest.approx(0.0)
    assert result["soc_wh"] == pytest.approx(0.0)
    assert result["soc_fraction"] == pytest.approx(0.0)


# 7. Battery at 100% SoC receives no more charge
def test_full_battery_no_charge():
    batt = BatteryModel(capacity_wh=CAPACITY, initial_soc_fraction=1.0)
    result = batt.step(power_draw_w=0.0, re_available_wh=10000.0)
    assert result["b_charge_wh"] == pytest.approx(0.0)


# 8. Battery at 0% SoC discharges nothing
def test_empty_battery_no_discharge():
    batt = BatteryModel(capacity_wh=CAPACITY, initial_soc_fraction=0.0)
    result = batt.step(power_draw_w=10000.0, re_available_wh=0.0)
    assert result["b_discharge_wh"] == pytest.approx(0.0)


# 9. reset() restores SoC regardless of prior steps
def test_reset_restores_state(battery):
    for _ in range(10):
        battery.step(150.0, 50.0)
    battery.reset(0.5)
    assert battery.soc_wh == pytest.approx(0.5 * CAPACITY)
    assert battery.soc_fraction == pytest.approx(0.5)


# 10. step() return dict has exactly the required keys
def test_step_return_keys(battery):
    result = battery.step(50.0, 80.0)
    assert set(result.keys()) == {"b_charge_wh", "b_discharge_wh", "soc_wh", "soc_fraction"}
