"""Unit tests for the pure-Python production schedule validator."""
from guards.schedule_validator import validate_production_schedule


def _baseline_demand_24h() -> dict[int, float]:
    """Representative 24-hour baseline town water demand (MLD)."""
    demands = {}
    for h in range(24):
        if 0 <= h < 6:
            demands[h] = 47.0
        elif 6 <= h < 14:
            demands[h] = 88.0
        elif 14 <= h < 19:
            demands[h] = 92.0
        else:
            demands[h] = 65.0
    return demands


def test_valid_schedule_passes_all_guards():
    schedule = (
        "00:00-06:00 at 110 MLD (off-peak $0.09/kWh), "
        "06:00-14:00 at 95 MLD, "
        "14:00-18:00 at 60 MLD (peak $0.48/kWh), "
        "19:00-24:00 at 85 MLD."
    )
    res = validate_production_schedule(
        proposed_schedule=schedule,
        current_volume_ml=240.0,  # 48%
        demand_by_hour=_baseline_demand_24h(),
        bloom_severity="NONE",
    )
    assert res["status"] == "VALID"
    assert res["is_valid"] is True
    assert len(res["blocks"]) == 4
    assert len(res["rejected_blocks"]) == 0
    assert all(b["status"] == "VALID" for b in res["blocks"])


def test_rejects_block_exceeding_desal_max_capacity():
    """Block '14:00-18:00 at 351 MLD' exceeds DESAL_MAX_CAPACITY_MLD (120 MLD)."""
    schedule = "00:00-06:00 at 90 MLD, 14:00-18:00 at 351 MLD."
    res = validate_production_schedule(
        proposed_schedule=schedule,
        current_volume_ml=240.0,
        demand_by_hour=_baseline_demand_24h(),
        bloom_severity="NONE",
    )
    assert res["status"] == "REJECTED_BY_CODE"
    assert res["is_valid"] is False
    assert len(res["rejected_blocks"]) == 1
    bad = res["rejected_blocks"][0]
    assert bad["window"] == "14:00-18:00"
    assert bad["status"] == "REJECTED_BY_CODE"
    assert any("120.0 MLD" in r for r in bad["reasons"])


def test_rejects_block_exceeding_algal_bloom_derating_limit():
    """MODERATE bloom derates max capacity to 60 MLD; 90 MLD block must be REJECTED_BY_CODE."""
    schedule = "00:00-06:00 at 50 MLD, 14:00-18:00 at 90 MLD."
    res = validate_production_schedule(
        proposed_schedule=schedule,
        current_volume_ml=240.0,
        demand_by_hour=_baseline_demand_24h(),
        bloom_severity="MODERATE",
    )
    assert res["status"] == "REJECTED_BY_CODE"
    assert res["bloom_max_safe_mld"] == 60.0
    assert len(res["rejected_blocks"]) == 1
    bad = res["rejected_blocks"][0]
    assert bad["window"] == "14:00-18:00"
    assert any("algal bloom derating" in r.lower() for r in bad["reasons"])


def test_rejects_block_breaching_35_percent_reservoir_floor():
    """Scenario 2 demand (351 MLD over 14:00-20:00) with 15 MLD throttle breaches 35% floor (31.2%)."""
    demand = _baseline_demand_24h()
    for h in range(14, 20):
        demand[h] = 351.0

    schedule = "00:00-06:00 at 110 MLD, 14:00-20:00 at 15 MLD."
    res = validate_production_schedule(
        proposed_schedule=schedule,
        current_volume_ml=240.0,  # 48%
        demand_by_hour=demand,
        bloom_severity="NONE",
    )
    assert res["status"] == "REJECTED_BY_CODE"
    assert len(res["rejected_blocks"]) == 1
    bad = res["rejected_blocks"][0]
    assert bad["window"] == "14:00-20:00"
    assert bad["min_projected_percent"] < 35.0
    assert any("SAFETY FLOOR BREACH" in r and "35.0%" in r for r in bad["reasons"])


def test_rejects_unparseable_vague_schedule():
    res = validate_production_schedule(
        proposed_schedule="Run desalination plant during off-peak hours to save money.",
        current_volume_ml=240.0,
        demand_by_hour=_baseline_demand_24h(),
    )
    assert res["status"] == "REJECTED_BY_CODE"
    assert res["is_valid"] is False
    assert len(res["rejected_blocks"]) == 1


def test_critical_bloom_caps_schedule_at_60_mld_not_zero():
    """CRITICAL bloom at Intake 2 caps output at 50% (60 MLD): 60 MLD passes while 80 MLD is REJECTED_BY_CODE."""
    schedule = "00:00-06:00 at 60 MLD, 14:00-18:00 at 80 MLD."
    res = validate_production_schedule(
        proposed_schedule=schedule,
        current_volume_ml=240.0,
        demand_by_hour=_baseline_demand_24h(),
        bloom_severity="CRITICAL",
    )
    assert res["bloom_max_safe_mld"] == 60.0
    assert res["status"] == "REJECTED_BY_CODE"
    assert res["blocks"][0]["window"] == "00:00-06:00"
    assert res["blocks"][0]["status"] == "VALID"
    assert res["blocks"][1]["window"] == "14:00-18:00"
    assert res["blocks"][1]["status"] == "REJECTED_BY_CODE"
    assert any("60.0 MLD" in r for r in res["blocks"][1]["reasons"])


def test_cumulative_drain_0_mld_schedule_rejected_from_1400():
    """
    From 48% at 14:00, '00:00-14:00 at 0 MLD, 14:00-18:00 at 60 MLD, 18:00-24:00 at 0 MLD'
    must carry levels forward cumulatively and be REJECTED_BY_CODE with first breach hour & %.
    """
    from database.seed_data import seed_database
    from tools.read_feeds import read_tariff_and_quality_feeds

    seed_database()
    seeded_demand = read_tariff_and_quality_feeds(current_hour=14)["hourly_demand_24h"]

    schedule = "00:00-14:00 at 0 MLD, 14:00-18:00 at 60 MLD, 18:00-24:00 at 0 MLD."
    res = validate_production_schedule(
        proposed_schedule=schedule,
        current_volume_ml=240.0,  # 48.0% at 14:00
        demand_by_hour=seeded_demand,
        bloom_severity="NONE",
        current_hour=14,
    )
    assert res["status"] == "REJECTED_BY_CODE"
    assert res["is_valid"] is False
    assert res["first_breach_hour"] is not None
    assert res["first_breach_percent"] is not None and res["first_breach_percent"] < 35.0
    assert len(res["cumulative_trajectory"]) == 25  # 14:00 start + 24 hourly steps


def test_cumulative_block_min_never_exceeds_previous_block_end_when_draining():
    """
    Whenever output < demand in consecutive blocks, each block's min_projected_percent
    must be <= its own start_projected_percent AND <= the previous block's end_projected_percent.
    """
    demand = {h: 120.0 for h in range(24)}  # constant 120 MLD demand > every block's output
    schedule = (
        "14:00-18:00 at 60 MLD, "
        "18:00-24:00 at 60 MLD, "
        "00:00-06:00 at 60 MLD, "
        "06:00-14:00 at 60 MLD."
    )
    res = validate_production_schedule(
        proposed_schedule=schedule,
        current_volume_ml=240.0,  # 48.0% at 14:00
        demand_by_hour=demand,
        bloom_severity="NONE",
        current_hour=14,
    )
    chrono = res["chronological_blocks"]
    assert [b["window"] for b in chrono] == ["14:00-18:00", "18:00-24:00", "00:00-06:00", "06:00-14:00"]

    prev_end = 48.0
    for blk in chrono:
        assert blk["start_projected_percent"] == prev_end
        assert blk["min_projected_percent"] <= blk["start_projected_percent"]
        assert blk["min_projected_percent"] <= prev_end
        assert blk["end_projected_percent"] < blk["start_projected_percent"]
        prev_end = blk["end_projected_percent"]


def test_cumulative_schedule_output_ge_demand_all_day_stays_valid():
    """A schedule whose output >= hourly demand across all 24 hours stays VALID and >= 48%."""
    demand = _baseline_demand_24h()  # max hourly demand is 92 MLD
    schedule = (
        "00:00-06:00 at 60 MLD, "
        "06:00-14:00 at 95 MLD, "
        "14:00-18:00 at 100 MLD, "
        "18:00-24:00 at 100 MLD."
    )
    res = validate_production_schedule(
        proposed_schedule=schedule,
        current_volume_ml=240.0,  # 48.0% at 14:00
        demand_by_hour=demand,
        bloom_severity="NONE",
        current_hour=14,
    )
    assert res["status"] == "VALID"
    assert res["is_valid"] is True
    assert res["first_breach_hour"] is None
    assert res["min_overall_percent"] >= 48.0
    assert all(b["status"] == "VALID" for b in res["blocks"])
    assert len(res["cumulative_trajectory"]) == 25


def test_uncovered_gap_hours_use_bloom_capped_setpoint_and_appear_as_gap():
    """Uncovered gap hours (18:00-19:00) use current_plant_output_mld capped by bloom limit and show as a gap."""
    schedule = (
        "00:00-06:00 at 60 MLD, "
        "06:00-14:00 at 60 MLD, "
        "14:00-18:00 at 60 MLD, "
        "19:00-24:00 at 60 MLD."
    )
    # Under CRITICAL bloom (50% = 60 MLD cap), a 100 MLD current setpoint is capped to 60 MLD during the 18:00-19:00 gap
    res_bloom = validate_production_schedule(
        proposed_schedule=schedule,
        current_volume_ml=240.0,
        demand_by_hour=_baseline_demand_24h(),
        bloom_severity="CRITICAL",
        current_hour=14,
        current_plant_output_mld=100.0,
    )
    assert len(res_bloom["gap_blocks"]) == 1
    gap = res_bloom["gap_blocks"][0]
    assert gap["is_gap"] is True
    assert gap["window"] == "18:00-19:00"
    assert gap["proposed_mld"] == 60.0
    assert "gap" in gap["gap_note"].lower()
    assert any(b.get("is_gap") and b["window"] == "18:00-19:00" for b in res_bloom["chronological_blocks"])

    # Without a bloom, the same 18:00-19:00 gap uses the full current plant setpoint (100 MLD)
    res_clean = validate_production_schedule(
        proposed_schedule=schedule,
        current_volume_ml=240.0,
        demand_by_hour=_baseline_demand_24h(),
        bloom_severity="NONE",
        current_hour=14,
        current_plant_output_mld=100.0,
    )
    assert len(res_clean["gap_blocks"]) == 1
    assert res_clean["gap_blocks"][0]["window"] == "18:00-19:00"
    assert res_clean["gap_blocks"][0]["proposed_mld"] == 100.0


def test_physics_hourly_reservoir_rise_never_exceeds_one_percent():
    """
    Physical invariant: max plant inflow is 120 MLD = 5 ML/hour into a 500 ML tank,
    so between any two consecutive hours the reservoir level can never rise by more than 1.0 percentage point
    (even with 0 MLD demand and an over-capacity schedule proposal).
    """
    zero_demand = {h: 0.0 for h in range(24)}
    for schedule in (
        "00:00-24:00 at 120 MLD",
        "00:00-24:00 at 351 MLD",  # over-capacity proposal clamped to physical 120 MLD max
    ):
        res = validate_production_schedule(
            proposed_schedule=schedule,
            current_volume_ml=240.0,  # 48.0%
            demand_by_hour=zero_demand,
            bloom_severity="NONE",
            current_hour=14,
        )
        traj = res["cumulative_trajectory"]
        assert len(traj) == 25
        for i in range(len(traj) - 1):
            rise = traj[i + 1]["level_percent"] - traj[i]["level_percent"]
            assert rise <= 1.0 + 1e-9, (
                f"Unphysical reservoir rise of +{rise:.4f}% between {traj[i]['hour_label']} and {traj[i+1]['hour_label']}"
            )
